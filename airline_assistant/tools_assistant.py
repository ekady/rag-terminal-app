from airline_assistant.db import get_connection


def get_destinations():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, destination FROM airline_prices;")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"id": x[0], "destination": x[1]} for x in result]


def get_ticket_price(destination: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, price FROM airline_prices WHERE LOWER(destination) = LOWER(%s);",
        (destination,),
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return {"id": result[0], "price": result[1]} if result else None


def get_current_bookings(passenger_name: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM bookings WHERE LOWER(passenger_name) = LOWER(%s);",
        (passenger_name,),
    )
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


def book_flight(passenger_name: str, destination: str):
    price = get_ticket_price(destination)
    if not price:
        return "Ticket not found"
    price_id = price["id"]
    conn = get_connection()
    cursor = conn.cursor()
    current_bookings = get_current_bookings(passenger_name)
    if len(current_bookings) >= 3:
        return "You have already booked a flight, maximum 3 flights per passenger"
    cursor.execute(
        "INSERT INTO bookings (passenger_name, airline_price_id, status) VALUES (%s, %s, 'booked');",
        (
            passenger_name,
            price_id,
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return "Flight booked successfully"


def cancel_booking(passenger_name: str, destination: str):
    price = get_ticket_price(destination)
    if not price:
        return "Ticket not found"
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM bookings WHERE passenger_name = %s AND airline_price_id = %s;",
        (passenger_name, price["id"]),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return "Flight cancelled successfully"
