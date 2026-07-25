import asyncio
import uuid
from core.database import AsyncSessionLocal
from core.models import User, Session

async def seed_db():
    async with AsyncSessionLocal() as db:
        # Create a dummy user
        dummy_user = User(id=uuid.uuid4(), email="test@test.com")
        db.add(dummy_user)
        
        # Create a dummy session tied to that user
        dummy_session = Session(id="test-session-123", user_id=dummy_user.id, title="Test Chat")
        db.add(dummy_session)
        
        await db.commit()
        print("Database Seeded! Use session_id: 'test-session-123'")

asyncio.run(seed_db())
