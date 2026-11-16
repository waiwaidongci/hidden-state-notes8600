import math,uuid
from datetime import datetime,timezone,timedelta
from sqlalchemy import select,or_
from fastapi import HTTPException
from src.models import ModelProfile,ObservationTrace,TrainingJob,TrainingBatch
from src.models.training import TrainingJob as _TJ
from src.models.training_batch import TrainingBatch as _TB
TJ=_TJ.__table__;TB=_TB.__table__
from src.utils.serialize import row
from src.services.parameters import normalize_parameters
from src.services.training_em import batch_expected_counts,m_step,sequence_log_likelihood,ImpossibleSequence
from .audit import record

LEASE_SECONDS=120
ACCEPT_SLACK=1e-12

def create_training(db,data,actor):
 profile=db.get(ModelProfile,data.profile_id)
 if not profile or profile.study_id!=data.study_id:raise HTTPException(422,'模型版本不属于该研究')
 ids=data.trace_ids
 if len(ids)!=len(set(ids)):raise HTTPException(422,'轨迹ID不能重复')
 traces=list(db.scalars(select(ObservationTrace).where(ObservationTrace.id.in_(ids),ObservationTrace.study_id==data.study_id).order_by(ObservationTrace.reference)))
 if len(traces)!=len(ids):raise HTTPException(422,'轨迹不属于该研究或不存在')
 if db.scalar(select(TrainingJob).where(TrainingJob.name==data.name)):raise HTTPException(409,'训练任务名已存在')
 if len(profile.symbols)!=len(profile.parameters['emission'][0]):raise HTTPException(422,'冻结版本发射列与观测码不一致')
 snapshot=row(profile);trace_snapshot=[row(t) for t in traces]
 job=TrainingJob(study_id=data.study_id,profile_id=profile.id,name=data.name,max_iterations=data.max_iterations,batch_size=data.batch_size,tolerance=float(data.tolerance),profile_snapshot=snapshot,trace_snapshot=trace_snapshot,current_parameters=snapshot['parameters'],status='queued',current_iteration=0,total_batches=0,total_log_likelihood=None,iteration_logs=[],created_by=actor)
 db.add(job);db.commit()
 try:
  from src.worker import run_training
  run_training.delay(job.id)
 except Exception:
  db.rollback();job=db.get(TrainingJob,job.id);job.status='failed';job.error='队列不可用，请稍后重建';db.commit();raise HTTPException(503,job.error)
 db.expire_all();job=db.get(TrainingJob,job.id)
 record(db,actor,'training.queued',{'job_id':job.id,'traces':len(traces),'iterations':data.max_iterations,'batch_size':data.batch_size})
 return row(job)

def _now():return datetime.now(timezone.utc)

def _aware(value):
 if value is None:return None
 return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

def _acquire_lease(db,job):
 owner=str(uuid.uuid4())
 deadline=_now()+timedelta(seconds=LEASE_SECONDS)
 free=or_(TJ.c.lease_expires_at.is_(None),TJ.c.lease_expires_at<_now())
 result=db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.status.in_(['queued','running']),free).values(lease_owner=owner,lease_expires_at=deadline,attempts=TJ.c.attempts+1,status='running'))
 db.commit()
 if result.rowcount!=1:return None
 db.expire_all()
 return owner

def _renew(db,job,owner):
 result=db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.lease_owner==owner).values(lease_expires_at=_now()+timedelta(seconds=LEASE_SECONDS)))
 db.commit()
 return result.rowcount==1

def _still_owner(db,job,owner):
 fresh=db.get(TrainingJob,job.id)
 return fresh is not None and fresh.lease_owner==owner and (fresh.lease_expires_at is None or _aware(fresh.lease_expires_at)>_now())

def _planned_batches(traces,batch_size):
 total=len(traces);return [(start,min(start+batch_size,total)) for start in range(0,total,batch_size)]

def _lookup(profile_snapshot):return {symbol:i for i,symbol in enumerate(profile_snapshot['symbols'])}

def _integer_traces(traces,lookup):
 return [[lookup[s] for s in t['tokens']] for t in sorted(traces,key=lambda t:t['reference'])]

def _ensure_iteration_batches(db,job,owner):
 existing=list(db.scalars(select(TrainingBatch).where(TrainingBatch.job_id==job.id,TrainingBatch.iteration==job.current_iteration).order_by(TrainingBatch.batch_index)))
 if existing:return existing
 traces=sorted(job.trace_snapshot,key=lambda t:t['reference']);plans=_planned_batches(traces,job.batch_size)
 for index,(start,end) in enumerate(plans):
  db.add(TrainingBatch(job_id=job.id,iteration=job.current_iteration,batch_index=index,trace_ids=[t['id'] for t in traces[start:end]],status='pending'))
 db.commit()
 if not _still_owner(db,job,owner):return []
 return list(db.scalars(select(TrainingBatch).where(TrainingBatch.job_id==job.id,TrainingBatch.iteration==job.current_iteration).order_by(TrainingBatch.batch_index)))

def _run_batch(db,job,batch,owner):
 if batch.status=='saved':return True
 if not _still_owner(db,job,owner):return False
 traces_by_id={t['id']:t for t in job.trace_snapshot}
 ordered=sorted((traces_by_id[i] for i in batch.trace_ids),key=lambda t:t['reference'])
 lookup=_lookup(job.profile_snapshot)
 integer_traces=_integer_traces(ordered,lookup)
 counts=batch_expected_counts(integer_traces,job.current_parameters)
 if not _still_owner(db,job,owner):return False
 result=db.execute(TB.update().where(TB.c.id==batch.id,TB.c.status=='pending').values(status='saved',pi_counts=counts['pi_counts'],transition_counts=counts['transition_counts'],emission_counts=counts['emission_counts'],log_likelihood=counts['log_likelihood'],token_count=counts['token_count'],written_by=owner))
 db.commit()
 if result.rowcount!=1:return False
 db.expire_all()
 return _still_owner(db,job,owner)

def _finish_iteration(db,job,owner):
 batches=list(db.scalars(select(TrainingBatch).where(TrainingBatch.job_id==job.id,TrainingBatch.iteration==job.current_iteration).order_by(TrainingBatch.batch_index)))
 if any(b.status!='saved' for b in batches) or not _still_owner(db,job,owner):return False
 K=len(job.current_parameters['pi']);M=len(job.current_parameters['emission'][0])
 pi_sum=[0.0]*K;A_sum=[[0.0]*K for _ in range(K)];B_sum=[[0.0]*M for _ in range(K)]
 for b in batches:
  pi_sum=[x+y for x,y in zip(pi_sum,b.pi_counts)]
  A_sum=[[x+y for x,y in zip(row_a,row_b)] for row_a,row_b in zip(A_sum,b.transition_counts)]
  B_sum=[[x+y for x,y in zip(row_a,row_b)] for row_a,row_b in zip(B_sum,b.emission_counts)]
 previous_log_likelihood=float(math.fsum(b.log_likelihood for b in batches))
 candidate=m_step(pi_sum,A_sum,B_sum,job.current_parameters)
 lookup=_lookup(job.profile_snapshot)
 integer_traces=_integer_traces(job.trace_snapshot,lookup)
 new_log_likelihood=float(math.fsum(sequence_log_likelihood(obs,candidate['pi'],candidate['transition'],candidate['emission']) for obs in integer_traces))
 accepted=new_log_likelihood>previous_log_likelihood+ACCEPT_SLACK
 next_parameters=candidate if accepted else job.current_parameters
 logs=list(job.iteration_logs)
 logs.append({'iteration':job.current_iteration+1,'previous_log_likelihood':previous_log_likelihood,'new_log_likelihood':new_log_likelihood,'accepted':accepted,'reason':'increase' if accepted else 'not_increase_rejected','batch_count':len(batches),'token_count':sum(b.token_count for b in batches)})
 result=db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.lease_owner==owner).values(current_iteration=job.current_iteration+1,current_parameters=next_parameters,total_log_likelihood=new_log_likelihood if accepted else previous_log_likelihood,iteration_logs=logs,total_batches=job.total_batches+len(batches)))
 db.commit()
 if result.rowcount!=1:return False
 db.expire_all()
 return _still_owner(db,job,owner)

def _publish(db,job,owner):
 if job.published_profile_id is not None:return True
 base=job.profile_snapshot
 normalized=normalize_parameters(job.current_parameters['pi'],job.current_parameters['transition'],job.current_parameters['emission'])
 name=f'{base["name"]}-EM训练-{job.id}'
 existing=db.scalar(select(ModelProfile).where(ModelProfile.name==name))
 if existing:
  db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.published_profile_id.is_(None)).values(published_profile_id=existing.id,status='completed',lease_owner=None,lease_expires_at=None,error=None))
  db.commit();return True
 profile=ModelProfile(study_id=job.study_id,name=name,version=f'em-iter{job.current_iteration}',states=base['states'],symbols=base['symbols'],parameters=normalized,normalization_max_delta=normalized['max_normalization_delta'],created_by='training.worker')
 db.add(profile);db.commit()
 profile=db.scalar(select(ModelProfile).where(ModelProfile.name==name))
 result=db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.lease_owner==owner,TJ.c.published_profile_id.is_(None)).values(published_profile_id=profile.id,status='completed',lease_owner=None,lease_expires_at=None,error=None))
 db.commit()
 if result.rowcount==0:
  db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.published_profile_id.is_(None)).values(published_profile_id=profile.id))
  db.commit()
 record(db,'training.worker','training.published',{'job_id':job.id,'profile_id':profile.id,'iterations':job.current_iteration})
 return True

def _fail(db,job,owner,error):
 result=db.execute(TJ.update().where(TJ.c.id==job.id,TJ.c.lease_owner==owner).values(status='failed',error=str(error)[:1000],lease_owner=None,lease_expires_at=None))
 db.commit()
 record(db,'training.worker','training.failed',{'job_id':job.id,'error':str(error)[:300]})
 return result.rowcount==1

def run_training_job(db,job_id):
 job=db.get(TrainingJob,job_id)
 if not job or job.status=='completed':return
 owner=_acquire_lease(db,job)
 if owner is None:return
 try:
  while True:
   db.expire_all()
   job=db.get(TrainingJob,job_id)
   if job.status!='running' or job.lease_owner!=owner:return
   if job.current_iteration>=job.max_iterations:
    _publish(db,job,owner);return
   if not _renew(db,job,owner):return
   db.expire_all();job=db.get(TrainingJob,job_id)
   batches=_ensure_iteration_batches(db,job,owner)
   if not batches:return
   for batch in batches:
    if not _renew(db,job,owner):return
    db.expire_all();job=db.get(TrainingJob,job_id)
    fresh=db.get(TrainingBatch,batch.id)
    if fresh.status=='saved':continue
    if not _run_batch(db,job,fresh,owner):return
   db.expire_all();job=db.get(TrainingJob,job_id)
   if not _finish_iteration(db,job,owner):return
   db.expire_all();job=db.get(TrainingJob,job_id)
   last=job.iteration_logs[-1]
   # 拒绝轮参数未变，重算结果必然相同，直接收敛；微小增益按容差收敛
   if not last['accepted'] or (last['new_log_likelihood']-last['previous_log_likelihood']<=job.tolerance):
    _publish(db,job,owner);return
 except ImpossibleSequence as error:
  job=db.get(TrainingJob,job_id)
  if job and job.lease_owner==owner:_fail(db,job,owner,f'不可能序列：{error}')
 except Exception as error:
  job=db.get(TrainingJob,job_id)
  if job and job.lease_owner==owner:_fail(db,job,owner,error)
