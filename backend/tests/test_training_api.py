"""训练 API 端到端：绕过 Celery 直接执行编排，验证鉴权、403、422 与快照冻结。"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import database as db_module
from src.config.database import get_db
from src.main import app
from src.middlewares import rate_limit as rate_limit_module
from src.models import SequenceStudy,ObservationTrace,ModelProfile,User
from src.services.parameters import normalize_parameters
from src.services import training as training_service
from src.utils.security import hash_password,issue_token
from src import worker as worker_module

@pytest.fixture()
def client_factory(monkeypatch):
 engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
 db_module.Base.metadata.create_all(engine)
 factory=sessionmaker(engine,expire_on_commit=False)
 monkeypatch.setattr(training_service,'record',lambda *a,**k:None)
 async def _passthrough(self,request,call_next):return await call_next(request)
 monkeypatch.setattr(rate_limit_module.RateLimitMiddleware,'dispatch',_passthrough)
 def override_get_db():
  db=factory()
  try:yield db
  finally:db.close()
 app.dependency_overrides[get_db]=override_get_db
 # 队列投递改为进程内同步执行，避免依赖 Redis/Celery
 class _SyncTask:
  def delay(self,job_id):training_service.run_training_job(factory(),job_id)
 monkeypatch.setattr(worker_module,'run_training',_SyncTask())
 with factory() as db:
  admin=User(email='admin@yanxi.demo',name='负责人',role='admin',password_hash=hash_password('pw'))
  viewer=User(email='viewer@yanxi.demo',name='观察员',role='viewer',password_hash=hash_password('pw'))
  db.add_all([admin,viewer]);db.flush()
  admin_id,viewer_id=admin.id,viewer.id
  study=SequenceStudy(name='API研究',filename='api.json',description='',symbols=['A','B']);db.add(study);db.flush()
  for i in range(7):db.add(ObservationTrace(study_id=study.id,reference=f'API-{i:03d}',tokens=(['A','B','A','B'] if i%2 else ['B','A','B','A'])))
  params=normalize_parameters([.6,.4],[[.7,.3],[.4,.6]],[[.6,.4],[.3,.7]])
  profile=ModelProfile(study_id=study.id,name='API固定模型',version='v1',states=['态甲','态乙'],symbols=['A','B'],parameters=params,normalization_max_delta=0.0,created_by='admin@yanxi.demo')
  db.add(profile);db.commit()
  trace_ids=[t.id for t in db.query(ObservationTrace).all()]
  ids=(study.id,profile.id,trace_ids,admin_id,viewer_id)
 yield TestClient(app),factory,ids
 app.dependency_overrides.clear()

def _auth(user_id):
 class _U:id=user_id
 return {'Authorization':'Bearer '+issue_token(_U())}

def test_training_end_to_end_and_freezes_snapshot(client_factory):
 client,factory,(study_id,profile_id,trace_ids,admin_id,_)=client_factory
 payload={'study_id':study_id,'profile_id':profile_id,'name':'API约束训练','trace_ids':trace_ids,'max_iterations':2,'batch_size':3,'tolerance':1e-6}
 r=client.post('/trainings',json=payload,headers=_auth(admin_id))
 assert r.status_code==200,r.text
 body=r.json();assert body['status']=='completed' and body['published_profile_id']
 detail=client.get(f"/trainings/{body['id']}",headers=_auth(admin_id)).json()
 assert len(detail['batches'])==6 and all(b['status']=='saved' for b in detail['batches'])
 assert [log['iteration'] for log in detail['iteration_logs']]==[1,2]
 with factory() as db:
  published=db.get(ModelProfile,body['published_profile_id'])
  assert published.id!=profile_id and published.study_id==study_id
  # 原始模型未被修改
  original=db.get(ModelProfile,profile_id)
  assert original.parameters['pi']==[.6,.4]

def test_viewer_cannot_create_training(client_factory):
 client,_,(study_id,profile_id,trace_ids,_,viewer_id)=client_factory
 r=client.post('/trainings',json={'study_id':study_id,'profile_id':profile_id,'name':'越权训练','trace_ids':trace_ids},headers=_auth(viewer_id))
 assert r.status_code==403

def test_requires_authentication(client_factory):
 client,_,(study_id,profile_id,trace_ids,_,_)=client_factory
 assert client.get('/trainings').status_code==401

def test_rejects_duplicate_trace_ids_and_foreign_profile(client_factory):
 client,_,(study_id,profile_id,trace_ids,admin_id,_)=client_factory
 r=client.post('/trainings',json={'study_id':study_id,'profile_id':profile_id,'name':'重复轨迹','trace_ids':[trace_ids[0],trace_ids[0]]},headers=_auth(admin_id))
 assert r.status_code==422
 r=client.post('/trainings',json={'study_id':999,'profile_id':profile_id,'name':'外研究版本','trace_ids':trace_ids},headers=_auth(admin_id))
 assert r.status_code==422

def test_duplicate_name_conflicts(client_factory):
 client,_,(study_id,profile_id,trace_ids,admin_id,_)=client_factory
 payload={'study_id':study_id,'profile_id':profile_id,'name':'同名训练','trace_ids':trace_ids}
 assert client.post('/trainings',json=payload,headers=_auth(admin_id)).status_code==200
 assert client.post('/trainings',json=payload,headers=_auth(admin_id)).status_code==409
