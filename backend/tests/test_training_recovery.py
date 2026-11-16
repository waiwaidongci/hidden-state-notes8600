"""实测批次中断恢复：在批次写入前后注入崩溃，重启后不重算已保存批次；
重复投递与旧执行（过期租约/非持有者）不重复累计、不覆盖新轮，最终只发布一个模型版本。"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime,timedelta,timezone

from src.config import database as db_module
from src.models import SequenceStudy,ObservationTrace,ModelProfile,TrainingJob,TrainingBatch
from src.services import training as training_service
from src.services.parameters import normalize_parameters
from src.utils.serialize import row

@pytest.fixture()
def session_factory(monkeypatch):
 engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
 db_module.Base.metadata.create_all(engine)
 factory=sessionmaker(engine,expire_on_commit=False)
 monkeypatch.setattr(training_service,'record',lambda *a,**k:None)
 with factory() as db:
  study=SequenceStudy(name='恢复研究',filename='crash.json',description='',symbols=['A','B'])
  db.add(study);db.flush()
  tokens_sets=[['A','B','A','A','B'],['B','B','A','B','A'],['A','A','B','B','A'],['B','A','A','B','B'],['A','B','B','A','A'],['B','A','B','A','B'],['A','A','A','B','B'],['B','B','A','A','B'],['A','B','A','B','A'],['B','A','B','B','A'],['A','A','B','A','B'],['B','B','B','A','A']]
  for i,tokens in enumerate(tokens_sets):db.add(ObservationTrace(study_id=study.id,reference=f'R-{i+1:03d}',tokens=tokens))
  params=normalize_parameters([.65,.35],[[.8,.2],[.35,.65]],[[.7,.3],[.25,.75]])
  profile=ModelProfile(study_id=study.id,name='初始固定模型',version='v1',states=['态甲','态乙'],symbols=['A','B'],parameters=params,normalization_max_delta=params['max_normalization_delta'],created_by='analyst@yanxi.demo')
  db.add(profile);db.commit()
  study_id,profile_id=study.id,profile.id
 return factory,study_id,profile_id

def _make_job(factory,study_id,profile_id,iterations=2,batch_size=4):
 with factory() as db:
  profile=db.get(ModelProfile,profile_id)
  traces=list(db.query(ObservationTrace).order_by(ObservationTrace.reference))
  job=TrainingJob(study_id=study_id,profile_id=profile_id,name=f'训练-{profile_id}-{iterations}-{batch_size}-{datetime.now().timestamp()}',max_iterations=iterations,batch_size=batch_size,tolerance=1e-6,profile_snapshot=row(profile),trace_snapshot=[row(t) for t in traces],current_parameters=profile.parameters,status='queued',current_iteration=0,total_batches=0,iteration_logs=[],created_by='analyst@yanxi.demo')
  db.add(job);db.commit();return job.id

def _complete_job(factory,job_id):
 training_service.run_training_job(factory(),job_id)
 with factory() as db:return db.get(TrainingJob,job_id)

def _simulate_dead_process(factory,job_id):
 # 进程崩溃后租约不会续约；等待租约超时后新进程才能接管
 with factory() as db:
  db.query(TrainingJob).filter_by(id=job_id).update({'lease_expires_at':datetime.now(timezone.utc)-timedelta(seconds=1)})
  db.commit()

def test_full_run_publishes_exactly_one_profile(session_factory):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 job=_complete_job(factory,job_id)
 assert job.status=='completed' and job.current_iteration==2 and job.published_profile_id
 with factory() as db:
  profiles=db.query(ModelProfile).filter(ModelProfile.study_id==study_id).all()
  assert len(profiles)==2
  assert len({j.published_profile_id for j in db.query(TrainingJob).all() if j.published_profile_id})==1
  batches=db.query(TrainingBatch).filter_by(job_id=job_id).all()
  assert len(batches)==6 and all(b.status=='saved' for b in batches)
  assert [l['iteration'] for l in job.iteration_logs]==[1,2]

class CrashAfterBatch(BaseException):pass

def test_crash_after_saved_batch_resumes_without_recompute(session_factory,monkeypatch):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 calls=[]
 state={'crashed':False}
 original=training_service.batch_expected_counts
 def instrumented(traces,parameters):
  result=original(traces,parameters);calls.append(len(traces));return result
 real_run_batch=training_service._run_batch
 def crashing_run_batch(db,job,batch,owner):
  ok=real_run_batch(db,job,batch,owner)
  if not state['crashed'] and job.current_iteration==0:
   state['crashed']=True
   raise CrashAfterBatch('进程在批次写入后崩溃')
  return ok
 monkeypatch.setattr(training_service,'batch_expected_counts',instrumented)
 monkeypatch.setattr(training_service,'_run_batch',crashing_run_batch)
 with pytest.raises(CrashAfterBatch):training_service.run_training_job(factory(),job_id)
 with factory() as db:
  job=db.get(TrainingJob,job_id);assert job.status=='running'
  saved=db.query(TrainingBatch).filter_by(job_id=job_id,iteration=0,status='saved').all();assert len(saved)==1
  assert len(calls)==1
 _simulate_dead_process(factory,job_id)
 # 恢复后不再崩溃；计数包装保留，已保存批次不会再次产生期望计数调用
 monkeypatch.setattr(training_service,'_run_batch',real_run_batch)
 job=_complete_job(factory,job_id)
 assert job.status=='completed'
 with factory() as db:
  batches=db.query(TrainingBatch).filter_by(job_id=job_id).order_by(TrainingBatch.iteration,TrainingBatch.batch_index).all()
  assert len(batches)==6 and all(b.status=='saved' for b in batches)
  # 首轮第一批已保存，重启后不重算：期望计数调用只覆盖其余5批
  assert len(calls)==6

def test_crash_before_batch_write_leaves_pending_and_resumes(session_factory,monkeypatch):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 state={'crashed':False}
 original=training_service.batch_expected_counts
 def crashing(traces,parameters):
  result=original(traces,parameters)
  if not state['crashed']:
   state['crashed']=True
   raise CrashAfterBatch('进程在批次写入前崩溃')
  return result
 monkeypatch.setattr(training_service,'batch_expected_counts',crashing)
 with pytest.raises(CrashAfterBatch):training_service.run_training_job(factory(),job_id)
 with factory() as db:
  pending=db.query(TrainingBatch).filter_by(job_id=job_id,iteration=0).all()
  assert all(b.status=='pending' for b in pending) and len(pending)==3
 monkeypatch.undo()
 _simulate_dead_process(factory,job_id)
 job=_complete_job(factory,job_id)
 assert job.status=='completed' and job.published_profile_id
 with factory() as db:
  assert all(b.status=='saved' for b in db.query(TrainingBatch).filter_by(job_id=job_id).all())

def test_duplicate_delivery_is_idempotent(session_factory):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 job=_complete_job(factory,job_id)
 first_published=job.published_profile_id
 # 完成后重复投递：直接返回，不新增任何批次或版本
 training_service.run_training_job(factory(),job_id)
 training_service.run_training_job(factory(),job_id)
 with factory() as db:
  job=db.get(TrainingJob,job_id)
  assert job.published_profile_id==first_published and job.status=='completed'
  assert db.query(TrainingBatch).filter_by(job_id=job_id).count()==6
  assert db.query(ModelProfile).filter(ModelProfile.study_id==study_id).count()==2
  assert job.attempts==1

def test_concurrent_second_worker_cannot_acquire_held_lease(session_factory):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 with factory() as db:
  job=db.get(TrainingJob,job_id);owner=training_service._acquire_lease(db,job)
 assert owner is not None
 with factory() as db:
  job=db.get(TrainingJob,job_id)
  assert training_service._acquire_lease(db,job) is None
  assert job.attempts==1

def test_expired_lease_is_reclaimed_and_old_owner_writes_are_rejected(session_factory):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 with factory() as db:
  job=db.get(TrainingJob,job_id);old_owner=training_service._acquire_lease(db,job)
  db.query(TrainingJob).filter_by(id=job_id).update({'lease_expires_at':datetime.now(timezone.utc)-timedelta(seconds=1)})
  db.commit()
 with factory() as db:
  job=db.get(TrainingJob,job_id);new_owner=training_service._acquire_lease(db,job)
 assert new_owner is not None and new_owner!=old_owner
 # 旧持有者的批次写入必须被租约归属校验拦截
 with factory() as db:
  job=db.get(TrainingJob,job_id)
  batches=training_service._ensure_iteration_batches(db,job,new_owner)
  assert len(batches)==3
  stale=db.get(TrainingBatch,batches[0].id)
  assert training_service._run_batch(db,job,stale,old_owner) is False
  assert db.get(TrainingBatch,stale.id).status=='pending'
 _simulate_dead_process(factory,job_id)
 job=_complete_job(factory,job_id)
 assert job.status=='completed' and job.published_profile_id

def test_iteration_logs_report_total_log_likelihood_each_round(session_factory):
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id,iterations=3,batch_size=6)
 job=_complete_job(factory,job_id)
 assert [entry['iteration'] for entry in job.iteration_logs]==[1,2,3]
 for entry in job.iteration_logs:
  assert isinstance(entry['previous_log_likelihood'],float)
  assert isinstance(entry['new_log_likelihood'],float)
 assert job.total_log_likelihood is not None

def test_tolerance_early_stop_publishes_before_max_iterations(session_factory):
 factory,study_id,profile_id=session_factory
 # 容差给到极大：首轮无论增益多少都满足收敛条件，应在第1轮后发布
 job_id=_make_job(factory,study_id,profile_id,iterations=5,batch_size=4)
 with factory() as db:
  db.query(TrainingJob).filter_by(id=job_id).update({'tolerance':1.0});db.commit()
 job=_complete_job(factory,job_id)
 assert job.status=='completed' and job.published_profile_id
 assert job.current_iteration==1 and len(job.iteration_logs)==1

def test_published_profile_is_usable_for_inference(session_factory):
 import math
 from src.services.hmm import infer_sequence
 factory,study_id,profile_id=session_factory
 job_id=_make_job(factory,study_id,profile_id)
 job=_complete_job(factory,job_id)
 with factory() as db:
  published=db.get(ModelProfile,job.published_profile_id)
  p=published.parameters
  result=infer_sequence([0,1,0,1],p['pi'],p['transition'],p['emission'])
  assert isinstance(result['log_likelihood'],float) and math.isfinite(result['log_likelihood'])
