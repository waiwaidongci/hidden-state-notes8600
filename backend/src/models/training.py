from sqlalchemy.orm import Mapped,mapped_column
from sqlalchemy import ForeignKey,String,JSON,Text,DateTime,Integer
from sqlalchemy.dialects.mysql import DOUBLE
from datetime import datetime,timezone
from src.config.database import Base
class TrainingJob(Base):
 __tablename__='training_jobs'
 id:Mapped[int]=mapped_column(primary_key=True)
 study_id:Mapped[int]=mapped_column(ForeignKey('sequence_studies.id'))
 profile_id:Mapped[int]=mapped_column(ForeignKey('model_profiles.id'))
 name:Mapped[str]=mapped_column(String(180),unique=True)
 max_iterations:Mapped[int]=mapped_column(Integer)
 batch_size:Mapped[int]=mapped_column(Integer)
 tolerance:Mapped[float]=mapped_column(DOUBLE(asdecimal=False))
 profile_snapshot:Mapped[dict]=mapped_column(JSON,nullable=True)
 trace_snapshot:Mapped[list]=mapped_column(JSON,nullable=True)
 current_parameters:Mapped[dict]=mapped_column(JSON,nullable=True)
 status:Mapped[str]=mapped_column(String(24),default='queued')
 current_iteration:Mapped[int]=mapped_column(Integer,default=0)
 total_batches:Mapped[int]=mapped_column(Integer,default=0)
 total_log_likelihood:Mapped[float|None]=mapped_column(DOUBLE(asdecimal=False),nullable=True)
 iteration_logs:Mapped[list]=mapped_column(JSON,default=list)
 published_profile_id:Mapped[int|None]=mapped_column(ForeignKey('model_profiles.id'),nullable=True)
 lease_owner:Mapped[str|None]=mapped_column(String(80),nullable=True)
 lease_expires_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
 attempts:Mapped[int]=mapped_column(Integer,default=0)
 error:Mapped[str|None]=mapped_column(Text,nullable=True)
 created_by:Mapped[str]=mapped_column(String(120))
 created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
