from sqlalchemy.orm import Mapped,mapped_column
from sqlalchemy import ForeignKey,String,JSON,DateTime,Integer,UniqueConstraint
from sqlalchemy.dialects.mysql import DOUBLE
from datetime import datetime,timezone
from src.config.database import Base
class TrainingBatch(Base):
 __tablename__='training_batches'
 __table_args__=(UniqueConstraint('job_id','iteration','batch_index',name='uq_training_batch_slot'),)
 id:Mapped[int]=mapped_column(primary_key=True)
 job_id:Mapped[int]=mapped_column(ForeignKey('training_jobs.id'))
 iteration:Mapped[int]=mapped_column(Integer)
 batch_index:Mapped[int]=mapped_column(Integer)
 trace_ids:Mapped[list]=mapped_column(JSON,nullable=True)
 status:Mapped[str]=mapped_column(String(24),default='pending')
 pi_counts:Mapped[list|None]=mapped_column(JSON,nullable=True)
 transition_counts:Mapped[list|None]=mapped_column(JSON,nullable=True)
 emission_counts:Mapped[list|None]=mapped_column(JSON,nullable=True)
 log_likelihood:Mapped[float|None]=mapped_column(DOUBLE(asdecimal=False),nullable=True)
 token_count:Mapped[int|None]=mapped_column(Integer,nullable=True)
 written_by:Mapped[str|None]=mapped_column(String(80),nullable=True)
 created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
