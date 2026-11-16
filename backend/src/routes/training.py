from fastapi import APIRouter,Depends,HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.controllers.dependencies import current_user,writers
from src.models import TrainingJob,TrainingBatch
from src.schemas.training import TrainingInput
from src.services.training import create_training
from src.utils.serialize import row
router=APIRouter()
@router.get('/trainings')
def trainings(db:Session=Depends(get_db),user=Depends(current_user)):
 return [row(j) for j in db.scalars(select(TrainingJob).order_by(TrainingJob.id.desc()))]
@router.get('/trainings/{id}')
def one(id:int,db:Session=Depends(get_db),user=Depends(current_user)):
 job=db.get(TrainingJob,id)
 if not job:raise HTTPException(404,'训练任务不存在')
 batches=list(db.scalars(select(TrainingBatch).where(TrainingBatch.job_id==id).order_by(TrainingBatch.iteration,TrainingBatch.batch_index)))
 return {**row(job),'batches':[row(b) for b in batches]}
@router.post('/trainings')
def create(data:TrainingInput,db:Session=Depends(get_db),user=Depends(writers)):
 return create_training(db,data,user.email)
