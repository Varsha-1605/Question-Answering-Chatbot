import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("POSTGRES_URL")
if not database_url:
    print("POSTGRES_URL is not set in .env")
    sys.exit(1)

try:
    with psycopg.connect(database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version(), current_database(), current_user, inet_server_addr()")
            version, database_name, user_name, server_ip = cursor.fetchone()

            cursor.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
            ssl_on = cursor.fetchone()[0]

    print("Connected to RDS")
    print(f"Database : {database_name}")
    print(f"User     : {user_name}")
    print(f"Server IP: {server_ip}")
    print(f"SSL      : {'on' if ssl_on else 'off'}")
    print(f"Version  : {version.split(',')[0]}")

except psycopg.OperationalError as error:
    print("Could not connect to RDS")
    print(error)
    sys.exit(1)