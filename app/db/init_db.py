"""Initialize database tables and seed product catalog data."""
import os
from datetime import datetime, timedelta
import random

from sqlalchemy.orm import Session

from app.db.session import engine, Base
from app.db.models import Product, Order


PRODUCTS = [
    # Electronics
    {"name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock_quantity": 150, "sku": "ELEC-001"},
    {"name": "Mechanical Keyboard", "category": "Electronics", "price": 89.99, "stock_quantity": 80, "sku": "ELEC-002"},
    {"name": "USB-C Hub 7-in-1", "category": "Electronics", "price": 45.00, "stock_quantity": 200, "sku": "ELEC-003"},
    {"name": "1080p Webcam", "category": "Electronics", "price": 79.00, "stock_quantity": 60, "sku": "ELEC-004"},
    {"name": "27-inch Monitor", "category": "Electronics", "price": 299.99, "stock_quantity": 30, "sku": "ELEC-005"},
    {"name": "Laptop Stand Adjustable", "category": "Electronics", "price": 35.00, "stock_quantity": 120, "sku": "ELEC-006"},
    # Office
    {"name": "Ballpoint Pens 10-pack", "category": "Office", "price": 8.99, "stock_quantity": 500, "sku": "OFFC-001"},
    {"name": "Notebook A5 Hardcover", "category": "Office", "price": 12.50, "stock_quantity": 300, "sku": "OFFC-002"},
    {"name": "Desk Organizer", "category": "Office", "price": 24.99, "stock_quantity": 90, "sku": "OFFC-003"},
    {"name": "Electric Stapler", "category": "Office", "price": 15.00, "stock_quantity": 70, "sku": "OFFC-004"},
    # Furniture
    {"name": "Ergonomic Office Chair", "category": "Furniture", "price": 349.99, "stock_quantity": 20, "sku": "FURN-001"},
    {"name": "Standing Desk 140cm", "category": "Furniture", "price": 499.99, "stock_quantity": 15, "sku": "FURN-002"},
    {"name": "Dual Monitor Arm", "category": "Furniture", "price": 75.00, "stock_quantity": 45, "sku": "FURN-003"},
    {"name": "Under-Desk Cable Tray", "category": "Furniture", "price": 22.00, "stock_quantity": 100, "sku": "FURN-004"},
    # Software
    {"name": "Antivirus Pro 1-Year", "category": "Software", "price": 39.99, "stock_quantity": 999, "sku": "SOFT-001"},
    {"name": "VPN License 1-Year", "category": "Software", "price": 59.99, "stock_quantity": 999, "sku": "SOFT-002"},
    {"name": "Cloud Backup 1TB/Year", "category": "Software", "price": 99.99, "stock_quantity": 999, "sku": "SOFT-003"},
]

CUSTOMERS = [
    "Alice Johnson", "Bob Smith", "Carol White", "David Brown",
    "Eve Davis", "Frank Miller", "Grace Lee", "Henry Wilson",
]

ORDERS_TEMPLATE = [
    # product_sku, quantity, customer, status, days_ago
    ("ELEC-001", 2, "Alice Johnson", "completed", 5),
    ("ELEC-002", 1, "Bob Smith", "completed", 12),
    ("ELEC-003", 3, "Carol White", "completed", 8),
    ("ELEC-004", 1, "David Brown", "pending", 2),
    ("ELEC-005", 1, "Eve Davis", "completed", 30),
    ("ELEC-006", 2, "Frank Miller", "completed", 20),
    ("OFFC-001", 5, "Grace Lee", "completed", 3),
    ("OFFC-002", 2, "Henry Wilson", "completed", 45),
    ("OFFC-003", 1, "Alice Johnson", "completed", 60),
    ("OFFC-004", 1, "Bob Smith", "cancelled", 7),
    ("FURN-001", 1, "Carol White", "completed", 90),
    ("FURN-002", 1, "David Brown", "pending", 1),
    ("FURN-003", 2, "Eve Davis", "completed", 15),
    ("FURN-004", 3, "Frank Miller", "completed", 22),
    ("SOFT-001", 1, "Grace Lee", "completed", 10),
    ("SOFT-002", 2, "Henry Wilson", "completed", 18),
    ("SOFT-003", 1, "Alice Johnson", "completed", 25),
    ("ELEC-001", 1, "Bob Smith", "completed", 35),
    ("ELEC-002", 2, "Carol White", "pending", 0),
    ("ELEC-003", 1, "David Brown", "completed", 50),
    ("OFFC-002", 4, "Eve Davis", "completed", 40),
    ("OFFC-001", 10, "Frank Miller", "completed", 55),
    ("FURN-001", 1, "Grace Lee", "cancelled", 65),
    ("ELEC-005", 2, "Henry Wilson", "completed", 70),
    ("SOFT-001", 3, "Alice Johnson", "completed", 80),
    ("SOFT-002", 1, "Bob Smith", "completed", 85),
    ("ELEC-004", 2, "Carol White", "completed", 75),
    ("FURN-002", 1, "David Brown", "cancelled", 4),
    ("OFFC-003", 2, "Eve Davis", "completed", 14),
    ("SOFT-003", 2, "Frank Miller", "pending", 1),
]


def init_db() -> None:
    """Create all tables (idempotent)."""
    # Ensure data directory exists
    os.makedirs("/app/data", exist_ok=True)

    Base.metadata.create_all(bind=engine)
    _seed_catalog()


def _seed_catalog() -> None:
    """Insert product catalog if not already seeded (idempotent)."""
    with Session(engine) as db:
        if db.query(Product).count() > 0:
            return  # Already seeded

        now_str = datetime.utcnow().isoformat()
        sku_to_id: dict[str, int] = {}

        for p in PRODUCTS:
            product = Product(
                name=p["name"],
                category=p["category"],
                price=p["price"],
                stock_quantity=p["stock_quantity"],
                sku=p["sku"],
                created_at=now_str,
            )
            db.add(product)
            db.flush()
            sku_to_id[p["sku"]] = product.id

        for sku, qty, customer, status, days_ago in ORDERS_TEMPLATE:
            product_id = sku_to_id[sku]
            # Find price for the product
            price = next(p["price"] for p in PRODUCTS if p["sku"] == sku)
            ordered_at = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
            order = Order(
                product_id=product_id,
                quantity=qty,
                total_price=round(price * qty, 2),
                customer_name=customer,
                status=status,
                ordered_at=ordered_at,
            )
            db.add(order)

        db.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
