import uuid
from datetime import datetime, UTC
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Index, JSON, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(500), default="新会话")
    status: Mapped[str] = mapped_column(String(20), default="active")
    last_seq: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC),
                                                 onupdate=lambda: datetime.now(UTC), index=True)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"))

    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    seq: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=True)
    tool_info: Mapped[dict] = mapped_column(JSON, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    __table_args__ = (
        Index("idx_msg_conv_seq", "conversation_id", "seq"),
    )


class SessionCheckpoint(Base):
    __tablename__ = "session_checkpoints"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    covers_from_seq: Mapped[int] = mapped_column(Integer)
    covers_to_seq: Mapped[int] = mapped_column(Integer)
    summary_json: Mapped[dict] = mapped_column(JSON)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ConsumerOffset(Base):
    __tablename__ = "consumer_offsets"
    consumer_group: Mapped[str] = mapped_column(String(128), primary_key=True)
    stream_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_acked_id: Mapped[str] = mapped_column(String(128), default="0-0")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC),
                                                 onupdate=lambda: datetime.now(UTC))
