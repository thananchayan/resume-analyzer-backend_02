from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "postgresql://postgres:root@localhost:5432/resume"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UserData(Base):
    __tablename__ = "user_data"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    job_role = Column(String)
    resume_score = Column(Integer)
    summary = Column(Text)
    recommended_skills = Column(Text)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Feedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    rating = Column(Integer)
    comment = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables if not exist
Base.metadata.create_all(bind=engine)
