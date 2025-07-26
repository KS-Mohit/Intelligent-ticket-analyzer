import re
import pandas as pd
import os

# --- Configuration ---
COMPLAINT_KEYWORDS = [
    'issue', 'problem', 'not working', 'broken', 'defective', 'faulty',
    'late', 'delay', 'wrong item', 'missing', 'error', 'fail', 'unable',
    'cannot', 'crashed'
]
DATE_REGEX = r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}(?:st|nd|rd|th)?,\s\d{4})'

# --- Extraction Functions ---
def extract_products(text, product_list):
    """Finds product names in text from a given list of products."""
    text_lower = text.lower()
    found_products = []

    # First, check for full product name matches (most accurate)
    for product in product_list:
        if product.lower() in text_lower:
            found_products.append(product)
    
    if found_products:
        return max(found_products, key=len)

    # If no full match, check for keywords from product names
    # Pad text with spaces to ensure boundary matching works correctly
    padded_text = f' {text_lower} '
    for product in product_list:
        for keyword in product.split():
            # Check if keyword is a meaningful word and exists in the text
            if len(keyword) > 2 and f' {keyword.lower()} ' in padded_text:
                 return product # Return the full product name it belongs to

    return None

def extract_dates(text):
    """Finds dates in text using regular expressions."""
    matches = re.findall(DATE_REGEX, text, re.IGNORECASE)
    return matches[0] if matches else None

def extract_complaints(text, keywords):
    """Finds complaint-related keywords in text."""
    found_keywords = [keyword for keyword in keywords if keyword.lower() in text.lower()]
    return list(set(found_keywords)) # Return unique keywords

def extract_entities(text, product_list):
    """Runs all extraction functions and returns a dictionary of entities."""
    product = extract_products(text, product_list)
    date = extract_dates(text)
    complaints = extract_complaints(text, COMPLAINT_KEYWORDS)

    return {
        "product": product,
        "date": date,
        "complaint_keywords": complaints
    }

# --- Demo / Testing Block ---
if __name__ == '__main__':
    print("--- Testing Entity Extractor ---")
    
    try:
        # Build path relative to this script
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
        # Use the correct filename as specified
        DATA_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Support_Tickets.xlsx')
        
        df = pd.read_excel(DATA_FILE_PATH)
        unique_products = df['product'].dropna().unique().tolist()
        print(f"Loaded {len(unique_products)} unique products for matching.")

    except FileNotFoundError:
        print(f"Error: Could not find the file at {DATA_FILE_PATH}")
        print("Using a dummy list for demo instead.")
        unique_products = ["SmartWatch V2", "SoundWave 300", "PhotoSnap Cam"]
    
    # Example tickets
    ticket1 = "My SmartWatch V2 is broken and the screen is not working. I bought it on 2025-07-20."
    ticket2 = "The delivery for my SoundWave 300 is late. Order placed on July 24th, 2025."
    ticket3 = "Just a quick question about my camera."

    print("\n--- Example 1 ---")
    print(f"Text: {ticket1}")
    print(f"Entities: {extract_entities(ticket1, unique_products)}")

    print("\n--- Example 2 ---")
    print(f"Text: {ticket2}")
    print(f"Entities: {extract_entities(ticket2, unique_products)}")

    print("\n--- Example 3 ---")
    print(f"Text: {ticket3}")
    print(f"Entities: {extract_entities(ticket3, unique_products)}")