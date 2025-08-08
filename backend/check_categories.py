#!/usr/bin/env python3
import os

import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_merchant_categories():
    try:
        # Database connection
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "fraud_detection"),
            user=os.getenv("DB_USER", "fraud_user"),
            password=os.getenv("DB_PASSWORD", "fraud_password"),
            port=os.getenv("DB_PORT", "5432"),
        )

        cursor = conn.cursor()

        print("🏪 Merchant Categories in Database:")
        print("=" * 50)

        # Get distinct merchant categories
        cursor.execute(
            """
            SELECT DISTINCT merchant_category, COUNT(*) as count
            FROM merchants 
            WHERE merchant_category IS NOT NULL
            GROUP BY merchant_category 
            ORDER BY merchant_category;
        """
        )

        categories = cursor.fetchall()

        if categories:
            for category, count in categories:
                print(f"📂 {category}: {count} merchants")
        else:
            print("❌ No categories found")

        print("\n" + "=" * 50)
        print(f"Total unique categories: {len(categories)}")

        # Also check transaction categories if they exist
        print("\n🔍 Transaction Categories (if any):")
        print("=" * 50)

        try:
            cursor.execute(
                """
                SELECT DISTINCT category, COUNT(*) as count
                FROM transactions 
                WHERE category IS NOT NULL
                GROUP BY category 
                ORDER BY category;
            """
            )

            trans_categories = cursor.fetchall()

            if trans_categories:
                for category, count in trans_categories:
                    print(f"📂 {category}: {count} transactions")
            else:
                print("❌ No transaction categories found")

        except Exception as e:
            print(f"ℹ️ No category column in transactions table: {e}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Database connection error: {e}")


if __name__ == "__main__":
    check_merchant_categories()
