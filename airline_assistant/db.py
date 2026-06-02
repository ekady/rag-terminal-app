import psycopg2
from config import Config

destination_prices = [
    {"destination": "Jakarta", "price": 1000000},
    {"destination": "Bali", "price": 1500000},
    {"destination": "Surabaya", "price": 2000000},
    {"destination": "Bandung", "price": 2500000},
    {"destination": "Yogyakarta", "price": 3000000},
    {"destination": "Semarang", "price": 3500000},
    {"destination": "Solo", "price": 4000000},
    {"destination": "Malang", "price": 4500000},
    {"destination": "Banyuwangi", "price": 5000000},
    {"destination": "Lombok", "price": 5500000},
]


def get_connection():
    return psycopg2.connect(
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT,
        dbname="airline_assistant",
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD,
    )


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS airline_prices (
        id SERIAL PRIMARY KEY,
        destination VARCHAR(255) NOT NULL,
        price DECIMAL(10, 2) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        id SERIAL PRIMARY KEY,
        passenger_name VARCHAR(255) NOT NULL,
        airline_price_id INT NOT NULL,
        status VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    for destination in destination_prices:
        cursor.execute(
            "INSERT INTO airline_prices (destination, price) VALUES (%s, %s);",
            (destination["destination"], destination["price"]),
        )
    conn.commit()
    cursor.close()
    conn.close()


def reset_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS airline_prices;")
    cursor.execute("DROP TABLE IF EXISTS bookings;")
    conn.commit()
    cursor.close()
    conn.close()
    init_db()


if __name__ == "__main__":
    init_db()
