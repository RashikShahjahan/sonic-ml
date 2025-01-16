import sqlite3
from datetime import datetime
from uuid import uuid4

def init_db():
    """Initialize SQLite database with tasks table"""
    conn = sqlite3.connect('sonic_ml.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks
        (id TEXT PRIMARY KEY, 
         command_type TEXT,
         status TEXT,
         created_at TIMESTAMP,
         updated_at TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

def update_task_status(task_id, status):
    """Update the status of a task"""
    conn = sqlite3.connect('sonic_ml.db')
    c = conn.cursor()
    c.execute('''
        UPDATE tasks 
        SET status = ?, updated_at = ?
        WHERE id = ?
    ''', (status, datetime.now(), task_id))
    conn.commit()
    conn.close()

def create_task(command_type):
    """Create a new task and return its ID"""
    task_id = str(uuid4())
    conn = sqlite3.connect('sonic_ml.db')
    c = conn.cursor()
    now = datetime.now()
    c.execute('''
        INSERT INTO tasks (id, command_type, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (task_id, command_type, 'running', now, now))
    conn.commit()
    conn.close()
    return task_id

def list_tasks():
    """List all tasks and their status"""
    conn = sqlite3.connect('sonic_ml.db')
    c = conn.cursor()
    c.execute('SELECT id, command_type, status FROM tasks')
    tasks = c.fetchall()
    conn.close()
    return tasks 