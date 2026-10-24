import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ecommerce_data() -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    
    n_rows = 150
    start_date = datetime(2025, 1, 1)
    
    categories = {
        "Technology": ["Phones", "Laptops", "Accessories", "Printers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Paper", "Storage", "Appliances", "Art"]
    }
    
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    segments = ["Consumer", "Corporate", "Small Business"]
    
    records = []
    for i in range(1, n_rows + 1):
        cat = random.choice(list(categories.keys()))
        sub_cat = random.choice(categories[cat])
        region = random.choice(regions)
        segment = random.choice(segments)
        days_offset = random.randint(0, 365)
        order_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        
        # Base pricing logic by category
        if cat == "Technology":
            base_price = random.uniform(250.0, 1200.0)
            quantity = random.randint(1, 5)
            discount = random.choice([0.0, 0.05, 0.1, 0.2, 0.3])
            margin = random.uniform(0.15, 0.35)
        elif cat == "Furniture":
            base_price = random.uniform(120.0, 650.0)
            quantity = random.randint(1, 7)
            discount = random.choice([0.0, 0.1, 0.15, 0.25, 0.4])
            margin = random.uniform(-0.05, 0.25)  # furniture sometimes operates at loss due to shipping
        else:
            base_price = random.uniform(15.0, 180.0)
            quantity = random.randint(1, 15)
            discount = random.choice([0.0, 0.0, 0.05, 0.1, 0.2])
            margin = random.uniform(0.20, 0.50)
            
        sales = round(base_price * quantity * (1.0 - discount), 2)
        profit = round(sales * margin, 2)
        
        # Introduce a few missing values intentionally for data quality testing
        if i in [14, 45, 89]:
            discount = np.nan
        returned = random.choice(["No", "No", "No", "No", "Yes"]) if i % 7 == 0 else "No"
        
        records.append({
            "OrderID": f"ORD-2025-{1000 + i}",
            "OrderDate": order_date,
            "Region": region,
            "CustomerSegment": segment,
            "Category": cat,
            "SubCategory": sub_cat,
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "Returned": returned
        })
        
    df = pd.DataFrame(records)
    # Add 2 duplicate rows intentionally to test duplicate detection
    df = pd.concat([df, df.iloc[[5, 23]]], ignore_index=True)
    return df


def generate_customer_churn_data() -> pd.DataFrame:
    random.seed(101)
    np.random.seed(101)
    
    n_rows = 140
    records = []
    
    genders = ["Male", "Female"]
    contracts = ["Month-to-month", "One year", "Two year"]
    internets = ["Fiber optic", "DSL", "No"]
    tech_supports = ["Yes", "No", "No internet service"]
    
    for i in range(1, n_rows + 1):
        gender = random.choice(genders)
        senior = random.choice([0, 0, 0, 1])
        partner = random.choice(["Yes", "No"])
        contract = random.choice(contracts)
        internet = random.choice(internets)
        
        if internet == "No":
            tech_support = "No internet service"
            monthly = round(random.uniform(19.0, 26.0), 2)
        elif internet == "Fiber optic":
            tech_support = random.choice(["Yes", "No"])
            monthly = round(random.uniform(70.0, 118.0), 2)
        else:
            tech_support = random.choice(["Yes", "No"])
            monthly = round(random.uniform(39.0, 75.0), 2)
            
        if contract == "Month-to-month":
            tenure = random.randint(1, 28)
            churn_prob = 0.55 if monthly > 70 else 0.35
        elif contract == "One year":
            tenure = random.randint(12, 48)
            churn_prob = 0.15
        else:
            tenure = random.randint(24, 72)
            churn_prob = 0.05
            
        total_charges = round(monthly * tenure * random.uniform(0.95, 1.05), 2)
        churn = "Yes" if random.random() < churn_prob else "No"
        
        # Missing total charges in a couple of records
        if i in [22, 67]:
            total_charges = np.nan
            
        records.append({
            "CustomerID": f"CUST-{7000 + i}",
            "Gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "TenureMonths": tenure,
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": tech_support,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
            "Churn": churn
        })
        
    df = pd.DataFrame(records)
    # Add 1 duplicate row
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    return df


def generate_titanic_data() -> pd.DataFrame:
    random.seed(777)
    np.random.seed(777)
    
    n_rows = 150
    records = []
    
    first_names_m = ["James", "John", "William", "George", "Charles", "Thomas", "Arthur", "Henry"]
    first_names_f = ["Mary", "Anna", "Margaret", "Elizabeth", "Alice", "Florence", "Helen", "Clara"]
    last_names = ["Smith", "Brown", "Johnson", "Williams", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Anderson"]
    embarked_ports = ["Southampton", "Cherbourg", "Queenstown"]
    
    for i in range(1, n_rows + 1):
        pclass = random.choices([1, 2, 3], weights=[0.24, 0.21, 0.55])[0]
        sex = random.choice(["male", "female"])
        
        if sex == "male":
            name = f"{random.choice(last_names)}, Mr. {random.choice(first_names_m)}"
            # Men had much lower survival rate
            surv_prob = 0.36 if pclass == 1 else (0.16 if pclass == 2 else 0.12)
        else:
            name = f"{random.choice(last_names)}, Mrs./Miss {random.choice(first_names_f)}"
            surv_prob = 0.96 if pclass == 1 else (0.90 if pclass == 2 else 0.50)
            
        survived = 1 if random.random() < surv_prob else 0
        
        # Age distribution
        if pclass == 1:
            age = round(random.gauss(38, 12), 1)
            fare = round(random.uniform(50.0, 260.0), 2)
        elif pclass == 2:
            age = round(random.gauss(30, 13), 1)
            fare = round(random.uniform(13.0, 45.0), 2)
        else:
            age = round(random.gauss(25, 11), 1)
            fare = round(random.uniform(7.2, 22.5), 2)
            
        age = max(1.0, min(80.0, age))
        
        # Missing values (typical for Titanic dataset)
        if i in [8, 19, 34, 52, 73, 91, 112, 135]:
            age = np.nan
            
        embarked = random.choices(embarked_ports, weights=[0.7, 0.2, 0.1])[0]
        if i in [30, 85]:
            embarked = np.nan
            
        sibsp = random.choices([0, 1, 2, 3], weights=[0.68, 0.23, 0.06, 0.03])[0]
        parch = random.choices([0, 1, 2], weights=[0.76, 0.16, 0.08])[0]
        
        records.append({
            "PassengerId": i,
            "Survived": survived,
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        })
        
    return pd.DataFrame(records)


def generate_housing_data() -> pd.DataFrame:
    random.seed(999)
    np.random.seed(999)
    
    n_rows = 130
    records = []
    
    neighborhoods = ["Downtown", "West End", "Highland Park", "Suburban Meadows", "Lakeside", "Eastside"]
    
    for i in range(1, n_rows + 1):
        hood = random.choice(neighborhoods)
        sqft = int(random.gauss(2100, 550))
        sqft = max(800, min(5000, sqft))
        
        bedrooms = max(1, min(6, int(sqft / 550) + random.choice([-1, 0, 1])))
        bathrooms = round(max(1.0, bedrooms * 0.75 + random.choice([0, 0.5, 1.0])), 1)
        year_built = random.randint(1975, 2024)
        has_pool = random.choices(["Yes", "No"], weights=[0.3, 0.7])[0]
        garage_spaces = random.choices([0, 1, 2, 3], weights=[0.1, 0.25, 0.55, 0.1])[0]
        
        # Price model
        base_rate = 220.0
        if hood == "Downtown":
            base_rate = 340.0
        elif hood == "Lakeside":
            base_rate = 380.0
        elif hood == "Highland Park":
            base_rate = 290.0
            
        price = sqft * base_rate + (bedrooms * 15000) + (bathrooms * 20000) + ((year_built - 1980) * 1200)
        if has_pool == "Yes":
            price += 35000
        price += random.gauss(0, 25000)
        price = round(max(120000, price), -2)
        
        days_on_market = max(2, int(random.expovariate(1/35)))
        
        records.append({
            "PropertyID": f"PROP-{5000 + i}",
            "Neighborhood": hood,
            "SquareFeet": sqft,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "YearBuilt": year_built,
            "GarageSpaces": garage_spaces,
            "HasPool": has_pool,
            "Price": price,
            "DaysOnMarket": days_on_market
        })
        
    return pd.DataFrame(records)


def generate_zomato_restaurants_data() -> pd.DataFrame:
    """Generate realistic Zomato restaurant dataset with cuisines, ratings, cost for two, locations, and reviews."""
    random.seed(333)
    np.random.seed(333)

    restaurant_templates = [
        ("Truffles", "Koramangala", "Cafe, Fast Food, Burger", "Cafe", 4.6, 900, 14850, "Yes", "No", "Legendary crispy burgers, Peri Peri fries, and dense Ferrero Rocher thickshakes."),
        ("Toit Brewpub", "Indiranagar", "Microbrewery, Continental, Pizza", "Microbrewery", 4.7, 2200, 16200, "No", "Yes", "Pioneering craft beers (Tin Man, Basmati Blonde) with wood-fired sourdough pizzas."),
        ("Meghana Foods", "Koramangala", "Biryani, Andhra, North Indian", "Casual Dining", 4.5, 750, 11400, "Yes", "No", "Signature spicy boneless chicken biryani with flavorful aromatic Andhra gravy."),
        ("Corner House Ice Cream", "Jayanagar", "Desserts, Ice Cream", "Dessert Parlor", 4.8, 350, 9200, "Yes", "No", "Iconic Death By Chocolate (DBC) and rich vanilla scoops smothered in hot fudge."),
        ("Vidyarthi Bhavan", "Malleshwaram", "South Indian, Traditional", "Quick Bites", 4.6, 200, 8500, "No", "No", "Heritage benne masala dosa with a crispy golden crust and rich filter coffee."),
        ("Empire Restaurant", "Church Street", "North Indian, Mughlai, Fast Food", "Casual Dining", 4.1, 800, 7800, "Yes", "No", "Late night favorite for coin parottas, ghee rice, and sizzling butter chicken."),
        ("Byg Brewski Brewing", "Bellandur", "Microbrewery, Continental, Asian", "Microbrewery", 4.6, 2400, 13100, "Yes", "Yes", "Massive open-air lakeside amphitheater ambience with world-class craft beer."),
        ("Glen's Bakehouse", "Indiranagar", "Bakery, Cafe, Desserts", "Bakery & Cafe", 4.5, 800, 6900, "Yes", "No", "Freshly baked red velvet mini cupcakes, artisanal cheesecakes, and iced lattes."),
        ("CTR Shri Sagar", "Malleshwaram", "South Indian, Breakfast", "Quick Bites", 4.7, 180, 7100, "No", "No", "Mouth-watering thick butter benne dosa served with three types of fresh chutney."),
        ("Punjab Grill", "Whitefield", "North Indian, Mughlai, Kebab", "Fine Dining", 4.4, 2500, 3900, "Yes", "Yes", "Royal North Indian dining with melt-in-mouth Galouti kebabs and rich Dal Makhani."),
        ("Onesta", "HSR Layout", "Italian, Pizza, Desserts", "Casual Dining", 4.0, 700, 5200, "Yes", "Yes", "Unlimited gourmet thin-crust pizza combos and mini dessert tarts at great value."),
        ("The Black Pearl", "Koramangala", "Barbecue, North Indian, Buffet", "Casual Dining", 4.3, 1600, 8900, "Yes", "Yes", "Pirate-themed ship buffet with live grilling skewers and endless coastal starters."),
        ("Nagarjuna", "Residency Road", "Andhra, Biryani, South Indian", "Casual Dining", 4.4, 900, 6200, "Yes", "No", "Authentic spicy Andhra meals served on banana leaf with gun powder and ghee."),
        ("Third Wave Coffee", "Koramangala", "Cafe, Coffee, Bakery", "Cafe", 4.4, 650, 4100, "Yes", "No", "Specialty pour-overs, artisanal flat whites, and flaky almond croissants."),
        ("Chianti", "Indiranagar", "Italian, Pizza, Wine Bar", "Fine Dining", 4.6, 2100, 4800, "No", "Yes", "Authentic Italian pastas, antipasti platters, and complimentary DIY bruschetta."),
        ("Chai Point", "Whitefield", "Tea, Fast Food, Snacks", "Quick Bites", 3.9, 300, 2400, "Yes", "No", "Quick ginger cardamom chai flasks with piping hot samosas and bun maska."),
        ("Leon Grill", "Koramangala", "Burger, Fast Food, Wraps", "Quick Bites", 4.2, 550, 3800, "Yes", "No", "Jumbo fried chicken burgers and peri peri loaded fries with spicy dips."),
        ("Windmills Craftworks", "Whitefield", "Microbrewery, Continental", "Microbrewery", 4.7, 2800, 6300, "No", "Yes", "Refined jazz theatre and library microbrewery with Belgian style ales and steaks."),
        ("The Hole in the Wall Cafe", "Koramangala", "Cafe, American, Breakfast", "Cafe", 4.5, 750, 7900, "Yes", "No", "All-day English breakfast, blueberry waffles, and fluffy scrambled eggs."),
        ("Brik Oven", "Church Street", "Italian, Pizza, Fast Food", "Casual Dining", 4.6, 1200, 5600, "Yes", "No", "Woodfired Neapolitan sourdough pizza with fresh buffalo mozzarella and sourdough crust."),
        ("Biergarten", "Koramangala", "Microbrewery, Continental, Asian", "Microbrewery", 4.5, 2100, 7400, "Yes", "Yes", "Lush outdoor green seating with Hefeweizen craft beer and spicy bao buns."),
        ("Yauatcha", "MG Road", "Chinese, Asian, Dim Sum", "Fine Dining", 4.5, 3100, 4200, "No", "Yes", "Michelin-starred style Cantonese dim sum, crispy duck rolls, and delicate macarons."),
        ("Social", "Church Street", "Modern Indian, Continental, Finger Food", "Pub & Bar", 4.3, 1500, 9800, "Yes", "Yes", "High-energy co-working pub with LIITs, loaded nachos, and quirky Indian fusion."),
        ("Asha Tiffins", "HSR Layout", "South Indian, Street Food", "Quick Bites", 4.5, 200, 4300, "Yes", "No", "Crispy ghee roast dosas and button idlis dunked in steaming hot sambar."),
        ("Farzi Cafe", "UB City", "Modern Indian, Fusion", "Fine Dining", 4.3, 2300, 5100, "Yes", "Yes", "Molecular gastronomy presentation with theatrical smoked cocktails and fusion tapas."),
        ("Smoor Chocolates", "Indiranagar", "Desserts, Bakery, Chocolates", "Dessert Parlor", 4.4, 900, 3100, "Yes", "No", "Decadent Belgian hot chocolate, handcrafted couverture truffles, and macarons."),
        ("Bastian", "St. Marks Road", "Seafood, Continental, Asian", "Fine Dining", 4.4, 3200, 2900, "No", "Yes", "Glamorous celebrity haunt serving butter garlic crab and burnt Basque cheesecake."),
        ("Oia Bangalore", "Hennur", "Mediterranean, Greek, European", "Casual Dining", 4.4, 2500, 4100, "Yes", "Yes", "Massive Santorini-inspired white architecture with mezze platters and cocktails."),
        ("Milano Ice Cream", "Indiranagar", "Desserts, Gelato, Italian", "Dessert Parlor", 4.7, 400, 4600, "Yes", "No", "Silky authentic Italian gelato with dark chocolate sorbet and pistachio crunch."),
        ("Mavalli Tiffin Room (MTR)", "Lalbagh", "South Indian, Heritage", "Quick Bites", 4.5, 300, 9400, "No", "No", "Iconic century-old institution famed for inventing the original Rava Idli.")
    ]

    locations = ["Koramangala", "Indiranagar", "MG Road", "Whitefield", "Jayanagar", "HSR Layout", "Church Street", "Malleshwaram", "Brigade Road", "BTM Layout", "JP Nagar", "Bellandur"]
    cuisines_list = [
        "Cafe, Fast Food, Burger", "Italian, Pizza, Pasta", "South Indian, Traditional", 
        "North Indian, Biryani, Mughlai", "Microbrewery, Continental, Pub", "Desserts, Ice Cream, Bakery", 
        "Chinese, Pan-Asian, Dim Sum", "Andhra, Biryani, Spicy", "Fast Food, Rolls, Grills", 
        "Continental, European, Grill", "Street Food, Chaat, Snacks"
    ]
    rest_types = ["Casual Dining", "Cafe", "Microbrewery", "Quick Bites", "Fine Dining", "Dessert Parlor", "Pub & Bar"]
    
    review_snippets = [
        "Outstanding food quality and attentive service. Will definitely visit again!",
        "Great lively atmosphere with vibrant music and delicious specialty drinks.",
        "Authentic regional flavours that remind you of traditional home cooking.",
        "Crispy starters and generous portions at very reasonable prices.",
        "Cosy ambience, perfect for weekend dates and coffee conversations.",
        "Best craft beer in the city paired with spicy bar nibbles.",
        "Decadent desserts and rich hot chocolate that melt in the mouth.",
        "Quick service and piping hot breakfast items with unlimited chutneys.",
        "Slightly crowded on Friday evenings, but the food is well worth the wait.",
        "Elegant decor and courteous staff making it ideal for family celebrations."
    ]

    records = []
    # Seed with the 30 iconic flagship restaurants
    for item in restaurant_templates:
        records.append({
            "Restaurant_Name": item[0],
            "Location": item[1],
            "Cuisine": item[2],
            "Rest_Type": item[3],
            "Rating": item[4],
            "Cost_For_Two": item[5],
            "Votes": item[6],
            "Online_Order": item[7],
            "Table_Booking": item[8],
            "Review_Highlight": item[9]
        })

    # Generate additional 220 realistic restaurant outlets
    names_prefix = ["The Urban", "Royal", "Spice", "Green", "Flavours of", "Brew &", "The Golden", "Little", "Cafe", "Master", "Daily", "Street", "Bite", "Saffron", "Basil", "Copper", "Coast"]
    names_suffix = ["Kitchen", "Bistro", "Treat", "Diner", "Spot", "Boutique", "Lounge", "Plate", "Corner", "Bakehouse", "Adda", "Junction", "House", "Villa", "Grill", "Hub"]

    for i in range(1, 221):
        name = f"{random.choice(names_prefix)} {random.choice(names_suffix)} #{i}"
        loc = random.choice(locations)
        cuisine = random.choice(cuisines_list)
        rtype = random.choice(rest_types)
        
        # Rating distribution centered around 4.1 with realistic spread
        rating = round(random.gauss(4.05, 0.38), 1)
        rating = max(3.1, min(4.9, rating))

        # Cost depends on type
        if rtype in ["Fine Dining", "Microbrewery"]:
            cost = int(random.gauss(2200, 450))
            cost = max(1400, min(3600, cost))
            table_book = random.choice(["Yes", "Yes", "No"])
        elif rtype in ["Cafe", "Casual Dining", "Pub & Bar"]:
            cost = int(random.gauss(850, 250))
            cost = max(450, min(1600, cost))
            table_book = random.choice(["Yes", "No", "No"])
        elif rtype in ["Dessert Parlor"]:
            cost = int(random.gauss(450, 150))
            cost = max(250, min(800, cost))
            table_book = "No"
        else: # Quick Bites
            cost = int(random.gauss(280, 80))
            cost = max(150, min(500, cost))
            table_book = "No"

        # Round cost to nearest 50
        cost = int(round(cost / 50.0) * 50)
        
        # Votes distribution
        votes = int(random.expovariate(1 / 1200)) + 60
        votes = min(15000, votes)
        
        online_ord = random.choice(["Yes", "Yes", "Yes", "No"])
        review = random.choice(review_snippets)

        # Introduce a few intentional missing values for real-world data health simulation
        if i in [18, 77, 142]:
            cost = np.nan
        if i in [33, 99]:
            rating = np.nan

        records.append({
            "Restaurant_Name": name,
            "Location": loc,
            "Cuisine": cuisine,
            "Rest_Type": rtype,
            "Rating": rating,
            "Cost_For_Two": cost,
            "Votes": votes,
            "Online_Order": online_ord,
            "Table_Booking": table_book,
            "Review_Highlight": review
        })

    df = pd.DataFrame(records)
    # Add 2 duplicate rows to test duplicate detection
    df = pd.concat([df, df.iloc[[12, 45]]], ignore_index=True)
    return df


def generate_100k_benchmark_data() -> pd.DataFrame:
    """Generate high-volume 100,000+ row dataset to test large scale analytics."""
    np.random.seed(42)
    n = 105000

    regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"], n, p=[0.32, 0.28, 0.24, 0.10, 0.06])
    categories = np.random.choice(["Technology", "Office Supplies", "Furniture", "Industrial", "Consumer Goods"], n, p=[0.25, 0.30, 0.20, 0.15, 0.10])
    channels = np.random.choice(["Online Store", "Retail Partner", "Enterprise Direct", "Mobile App"], n)
    units = np.random.randint(1, 12, n)
    prices = np.random.uniform(15.0, 480.0, n).round(2)
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.25, 0.30], n, p=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    ratings = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.08, 0.22, 0.35, 0.30])
    
    revenue = (units * prices * (1.0 - discounts)).round(2)
    margin = np.random.uniform(0.08, 0.42, n)
    profit = (revenue * margin).round(2)

    df = pd.DataFrame({
        "TransactionID": [f"TX-{100000 + i}" for i in range(1, n + 1)],
        "Region": regions,
        "Category": categories,
        "Channel": channels,
        "Quantity": units,
        "UnitPrice": prices,
        "DiscountPct": discounts,
        "Revenue": revenue,
        "Profit": profit,
        "CustomerRating": ratings
    })
    
    # Introduce ~0.5% missing values randomly for real-world simulation
    mask_discount = np.random.rand(n) < 0.005
    df.loc[mask_discount, "DiscountPct"] = np.nan
    
    return df


SAMPLE_REGISTRY = {
    "zomato_restaurants": {
        "name": "Zomato Bangalore Restaurant Intelligence",
        "description": "Comprehensive restaurant dataset exploring ratings, cuisines, dining hotspots, cost for two, customer votes, and dining types.",
        "category": "Hospitality & Food Analytics",
        "tags": ["Zomato", "Ratings", "Cuisines", "Pricing", "Locations", "Bangalore"],
        "generator": generate_zomato_restaurants_data,
        "sample_questions": [
            "Which cuisines have the highest average rating?",
            "What is the correlation between cost for two and restaurant rating?",
            "Which locations have the highest concentration of restaurants?",
            "Do restaurants with table booking have higher ratings?",
            "Show me the top 10 restaurants with the highest votes"
        ]
    },
    "ecommerce_sales": {
        "name": "Global Retail & E-Commerce Sales",
        "description": "Multi-region transaction data with sales, profit margins, discounts, customer segments, and returned orders.",
        "category": "Retail & Business",
        "tags": ["Sales", "Profits", "Discounts", "Segments", "Time-Series"],
        "generator": generate_ecommerce_data,
        "sample_questions": [
            "Which product category has the highest average profit?",
            "What is the correlation between discount and profit?",
            "Show me sales and profit breakdown by region",
            "What percentage of orders were returned?",
            "Who are the top customer segments driving revenue?"
        ]
    },
    "bigdata_100k": {
        "name": "Big Data Benchmark (100,000+ Rows)",
        "description": "Enterprise-scale 105,000-row transaction ledger testing rapid data ingestion, vectorized statistics, outlier radar, and sub-second analytics.",
        "category": "High-Volume Data Science",
        "tags": ["100k+ Rows", "Big Data", "Benchmark", "High Volume", "Vectorized"],
        "generator": generate_100k_benchmark_data,
        "sample_questions": [
            "What is the total revenue and profit across all 100,000+ transactions?",
            "Which region generated the highest total revenue?",
            "What is the correlation between UnitPrice and Revenue?",
            "What are the top 5 highest-revenue transactions?",
            "Are there any outliers in Revenue or Profit?"
        ]
    },
    "customer_churn": {
        "name": "Telecom Customer Churn & Retention",
        "description": "Subscriber account data examining contract types, monthly charges, tenure, and churn risk factors.",
        "category": "Customer Analytics",
        "tags": ["Churn", "Tenure", "Contracts", "Pricing", "Classification"],
        "generator": generate_customer_churn_data,
        "sample_questions": [
            "What is the overall churn rate of customers?",
            "Do customers with month-to-month contracts churn more often?",
            "What is the average monthly charge for churned vs retained customers?",
            "How does customer tenure affect churn likelihood?",
            "Which internet service type has the highest churn?"
        ]
    },
    "titanic_survival": {
        "name": "Titanic Passenger Demographics & Survival",
        "description": "Historical passenger manifest exploring survival rates across ticket classes, gender, age, and passenger fares.",
        "category": "Historical & Social Science",
        "tags": ["Survival", "Demographics", "Pclass", "Missing Values", "Benchmark"],
        "generator": generate_titanic_data,
        "sample_questions": [
            "What was the survival rate of female vs male passengers?",
            "How did ticket class (Pclass) influence survival chances?",
            "What is the average age of passengers who survived vs perished?",
            "Which boarding port (Embarked) had the highest number of passengers?",
            "Are there missing values in the age column?"
        ]
    },
    "housing_market": {
        "name": "Metropolitan Real Estate & Housing Prices",
        "description": "Property market dataset with square footage, bedrooms, year built, amenities, neighborhood, and listing prices.",
        "category": "Real Estate & Economics",
        "tags": ["Real Estate", "Pricing", "SquareFeet", "Correlations", "Outliers"],
        "generator": generate_housing_data,
        "sample_questions": [
            "What is the correlation between square feet and property price?",
            "Which neighborhood has the highest median house price?",
            "How does having a pool impact the average home price?",
            "What is the distribution of days on market?",
            "Show me the top 5 most expensive properties"
        ]
    }
}

