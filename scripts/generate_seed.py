#!/usr/bin/env python3
"""Build the offline Canadian grocery seed: products, recipes, baskets, labels.

Prices, stock, and flyer deals are synthetic. The order file uses the Instacart
shape (order id, sku list, train/test split) so the same trainer runs on a
Kaggle extract produced by scripts/import_instacart.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "seed"

# id, name, aisle, group, kcal, protein, carbs, fat, cents/100g, pack_g, tags, allergens
# tags: v vegan, V vegetarian, g gluten-free, P pork, B beef, F fish, S shellfish,
#       D dairy, E egg, N nuts, Y soy
ROWS = [
    ("onion", "Onion", "Produce", "allium", 40, 1.1, 9.3, 0.1, 22, 900, "vg", ""),
    ("green_onion", "Green Onion", "Produce", "allium", 32, 1.8, 7.3, 0.2, 90, 80, "vg", ""),
    ("garlic", "Garlic", "Produce", "allium", 149, 6.4, 33, 0.5, 80, 80, "vg", ""),
    ("ginger", "Ginger", "Produce", "aromatic", 80, 1.8, 18, 0.8, 140, 80, "vg", ""),
    ("broccoli", "Broccoli", "Produce", "crucifer", 34, 2.8, 7, 0.4, 85, 300, "vg", ""),
    ("broccoli_raab", "Broccoli Raab", "Produce", "crucifer", 22, 3.2, 2.8, 0.5, 160, 250, "vg", ""),
    ("spinach", "Spinach", "Produce", "greens", 23, 2.9, 3.6, 0.4, 220, 142, "vg", ""),
    ("kale", "Kale", "Produce", "greens", 35, 2.9, 4.4, 1.5, 180, 142, "vg", ""),
    ("lettuce", "Romaine Lettuce", "Produce", "greens", 17, 1.2, 3.3, 0.3, 70, 300, "vg", ""),
    ("arugula", "Arugula", "Produce", "greens", 25, 2.6, 3.7, 0.7, 240, 142, "vg", ""),
    ("tomato", "Tomato", "Produce", "nightshade", 18, 0.9, 3.9, 0.2, 45, 680, "vg", ""),
    ("bell_pepper", "Bell Pepper", "Produce", "nightshade", 26, 1.0, 6, 0.3, 80, 300, "vg", ""),
    ("zucchini", "Zucchini", "Produce", "squash", 17, 1.2, 3.1, 0.3, 55, 400, "vg", ""),
    ("eggplant", "Eggplant", "Produce", "nightshade", 25, 1.0, 6, 0.2, 70, 400, "vg", ""),
    ("carrot", "Carrot", "Produce", "root", 41, 0.9, 10, 0.2, 28, 900, "vg", ""),
    ("potato", "Potato", "Produce", "root", 77, 2.0, 17, 0.1, 20, 1500, "vg", ""),
    ("sweet_potato", "Sweet Potato", "Produce", "root", 86, 1.6, 20, 0.1, 30, 900, "vg", ""),
    ("beets", "Beets", "Produce", "root", 43, 1.6, 10, 0.2, 40, 680, "vg", ""),
    ("mushroom", "Cremini Mushrooms", "Produce", "veg", 22, 2.5, 3.3, 0.1, 120, 227, "vg", ""),
    ("avocado", "Avocado", "Produce", "fruit", 160, 2.0, 8.5, 15, 180, 400, "vg", ""),
    ("lemon", "Lemon", "Produce", "fruit", 29, 1.1, 9, 0.3, 55, 400, "vg", ""),
    ("lime", "Lime", "Produce", "fruit", 30, 0.7, 11, 0.2, 60, 300, "vg", ""),
    ("cilantro", "Cilantro", "Produce", "herb", 23, 2.1, 3.7, 0.5, 200, 30, "vg", ""),
    ("parsley", "Parsley", "Produce", "herb", 36, 3.0, 6.3, 0.8, 180, 30, "vg", ""),
    ("basil", "Basil", "Produce", "herb", 23, 3.2, 2.6, 0.6, 250, 20, "vg", ""),
    ("cucumber", "Cucumber", "Produce", "veg", 15, 0.7, 3.6, 0.1, 35, 400, "vg", ""),
    ("cabbage", "Green Cabbage", "Produce", "crucifer", 25, 1.3, 6, 0.1, 25, 800, "vg", ""),
    ("celery", "Celery", "Produce", "veg", 14, 0.7, 3, 0.2, 40, 450, "vg", ""),
    ("corn", "Corn", "Produce", "veg", 86, 3.3, 19, 1.2, 40, 400, "vg", ""),
    ("banana", "Banana", "Produce", "fruit", 89, 1.1, 23, 0.3, 18, 700, "vg", ""),
    ("apple", "Apple", "Produce", "fruit", 52, 0.3, 14, 0.2, 35, 900, "vg", ""),
    ("peas", "Green Peas", "Frozen", "veg", 77, 5.0, 14, 0.4, 45, 500, "vg", ""),
    ("chicken_breast", "Chicken Breast", "Meat", "poultry", 120, 31, 0, 2.6, 190, 600, "g", ""),
    ("chicken_thigh", "Chicken Thighs", "Meat", "poultry", 177, 24, 0, 8, 125, 700, "g", ""),
    ("chicken_ground", "Ground Chicken", "Meat", "poultry", 143, 23, 0, 6, 150, 450, "g", ""),
    ("turkey_breast", "Turkey Breast", "Meat", "poultry", 135, 29, 0, 1, 200, 500, "g", ""),
    ("turkey_ground", "Ground Turkey", "Meat", "poultry", 150, 27, 0, 5, 160, 450, "g", ""),
    ("ground_beef", "Ground Beef", "Meat", "beef", 215, 26, 0, 12, 170, 450, "Bg", ""),
    ("stewing_beef", "Stewing Beef", "Meat", "beef", 170, 28, 0, 6, 190, 500, "Bg", ""),
    ("sirloin", "Beef Sirloin", "Meat", "beef", 180, 27, 0, 8, 280, 400, "Bg", ""),
    ("pork_chop", "Pork Chops", "Meat", "pork", 170, 22, 0, 9, 140, 500, "Pg", ""),
    ("bacon", "Bacon", "Meat", "pork", 417, 12, 1.4, 40, 180, 375, "Pg", ""),
    ("ground_pork", "Ground Pork", "Meat", "pork", 260, 17, 0, 21, 130, 450, "Pg", ""),
    ("italian_sausage", "Italian Pork Sausage", "Meat", "pork", 300, 16, 2, 26, 150, 450, "P", ""),
    ("chicken_nuggets", "Chicken Nuggets", "Frozen", "prepared", 270, 14, 16, 16, 120, 700, "E", "wheat,egg"),
    ("chicken_soup", "Chicken Noodle Soup", "Pantry", "prepared", 45, 2.5, 6, 1.2, 40, 540, "", "wheat"),
    ("salmon", "Atlantic Salmon", "Seafood", "finfish", 208, 20, 0, 13, 340, 300, "Fg", "fish"),
    ("cod", "Cod Fillets", "Seafood", "finfish", 82, 18, 0, 0.7, 220, 400, "Fg", "fish"),
    ("shrimp", "Shrimp", "Seafood", "shellfish", 85, 24, 0.2, 0.5, 300, 300, "Sg", "shellfish"),
    ("tuna", "Canned Tuna", "Pantry", "finfish", 116, 25, 0, 1, 129, 170, "Fg", "fish"),
    ("tofu", "Firm Tofu", "Produce", "plant_protein", 144, 17, 2.8, 8.7, 80, 350, "vgY", "soy"),
    ("milk_2", "2% Milk", "Dairy", "milk", 50, 3.3, 4.8, 2.0, 18, 2000, "VDg", "milk"),
    ("milk_skim", "Skim Milk", "Dairy", "milk", 34, 3.4, 5.0, 0.1, 17, 2000, "VDg", "milk"),
    ("oat_milk", "Oat Milk", "Dairy", "milk", 45, 1.0, 7, 1.5, 28, 1800, "vg", ""),
    ("yogurt_greek", "Greek Yogurt", "Dairy", "yogurt", 97, 10, 3.6, 5, 70, 650, "VDg", "milk"),
    ("yogurt_plain", "Plain Yogurt", "Dairy", "yogurt", 61, 3.5, 4.7, 3.3, 45, 650, "VDg", "milk"),
    ("cheddar", "Cheddar Cheese", "Dairy", "cheese", 403, 25, 1.3, 33, 180, 320, "VDg", "milk"),
    ("mozzarella", "Mozzarella", "Dairy", "cheese", 280, 28, 3.1, 17, 160, 250, "VDg", "milk"),
    ("feta", "Feta Cheese", "Dairy", "cheese", 264, 14, 4.1, 21, 250, 200, "VDg", "milk"),
    ("parmesan", "Parmesan", "Dairy", "cheese", 420, 38, 4, 28, 280, 200, "VDg", "milk"),
    ("butter", "Butter", "Dairy", "fat", 717, 0.9, 0.1, 81, 140, 454, "VDg", "milk"),
    ("eggs", "Eggs", "Dairy", "eggs", 143, 13, 0.7, 10, 70, 600, "VEg", "egg"),
    ("rice", "Long Grain Rice", "Pantry", "grain", 365, 7, 80, 0.7, 40, 900, "vg", ""),
    ("quinoa", "Quinoa", "Pantry", "grain", 368, 14, 64, 6, 90, 700, "vg", ""),
    ("pasta", "Spaghetti", "Pantry", "grain", 371, 13, 75, 1.5, 35, 450, "v", "wheat"),
    ("bread", "White Bread", "Bakery", "bakery", 265, 9, 49, 3.2, 40, 570, "V", "wheat"),
    ("tortilla", "Flour Tortillas", "Bakery", "bakery", 310, 8, 52, 8, 50, 340, "v", "wheat"),
    ("flour", "All-Purpose Flour", "Pantry", "baking", 364, 10, 76, 1, 20, 1000, "v", "wheat"),
    ("oats", "Rolled Oats", "Pantry", "grain", 379, 13, 68, 7, 30, 1000, "vg", ""),
    ("canned_tomato", "Canned Tomatoes", "Pantry", "nightshade", 32, 1.6, 7, 0.3, 28, 796, "vg", ""),
    ("black_beans", "Black Beans", "Pantry", "plant_protein", 132, 8.9, 24, 0.5, 32, 540, "vg", ""),
    ("chickpeas", "Chickpeas", "Pantry", "plant_protein", 164, 8.9, 27, 2.6, 30, 540, "vg", ""),
    ("lentils", "Lentils", "Pantry", "plant_protein", 352, 25, 63, 1.1, 40, 900, "vg", ""),
    ("kidney_beans", "Kidney Beans", "Pantry", "plant_protein", 127, 8.7, 23, 0.5, 30, 540, "vg", ""),
    ("coconut_milk", "Coconut Milk", "Pantry", "canned", 197, 2.0, 3, 21, 55, 400, "vg", ""),
    ("peanut_butter", "Peanut Butter", "Pantry", "nut", 588, 25, 20, 50, 70, 500, "vgN", "peanut"),
    ("soy_sauce", "Soy Sauce", "Pantry", "condiment", 53, 8, 5, 0.1, 80, 296, "vY", "soy,wheat"),
    ("tamari", "Tamari", "Pantry", "condiment", 60, 10, 6, 0.1, 140, 250, "vgY", "soy"),
    ("miso", "Miso Paste", "Pantry", "condiment", 198, 12, 26, 6, 160, 200, "vgY", "soy"),
    ("olive_oil", "Olive Oil", "Pantry", "oil", 884, 0, 0, 100, 120, 500, "vg", ""),
    ("canola_oil", "Canola Oil", "Pantry", "oil", 884, 0, 0, 100, 50, 946, "vg", ""),
    ("sesame_oil", "Sesame Oil", "Pantry", "oil", 884, 0, 0, 100, 180, 185, "vg", "sesame"),
    ("vinegar", "White Vinegar", "Pantry", "condiment", 18, 0, 0.04, 0, 20, 1000, "vg", ""),
    ("honey", "Honey", "Pantry", "sweetener", 304, 0.3, 82, 0, 90, 500, "Vg", ""),
    ("sugar", "Granulated Sugar", "Pantry", "sweetener", 387, 0, 100, 0, 18, 1000, "vg", ""),
    ("powdered_sugar", "Powdered Sugar", "Pantry", "sweetener", 389, 0, 100, 0, 28, 500, "vg", ""),
    ("baking_soda", "Baking Soda", "Pantry", "baking", 0, 0, 0, 0, 20, 500, "vg", ""),
    ("salt", "Salt", "Pantry", "spice", 0, 0, 0, 0, 16, 500, "vg", ""),
    ("black_pepper", "Black Pepper", "Pantry", "spice", 251, 10, 64, 3.3, 400, 50, "vg", ""),
    ("chili_flakes", "Chili Flakes", "Pantry", "spice", 318, 12, 56, 6, 300, 40, "vg", ""),
    ("cumin", "Cumin", "Pantry", "spice", 375, 18, 44, 22, 280, 45, "vg", ""),
    ("paprika", "Paprika", "Pantry", "spice", 282, 14, 54, 13, 260, 50, "vg", ""),
    ("curry_powder", "Curry Powder", "Pantry", "spice", 325, 14, 58, 14, 240, 50, "vg", ""),
    ("turmeric", "Turmeric", "Pantry", "spice", 312, 10, 67, 3.2, 300, 45, "vg", ""),
    ("stock_chicken", "Chicken Broth", "Pantry", "broth", 8, 0.8, 0.6, 0.2, 22, 900, "g", ""),
    ("stock_veg", "Vegetable Broth", "Pantry", "broth", 7, 0.4, 0.8, 0.1, 22, 900, "vg", ""),
    ("salsa", "Salsa", "Pantry", "condiment", 36, 1.5, 7, 0.2, 45, 418, "vg", ""),
    ("mayonnaise", "Mayonnaise", "Pantry", "condiment", 680, 1, 0.6, 75, 50, 445, "VEg", "egg"),
    ("dijon", "Dijon Mustard", "Pantry", "condiment", 66, 4, 6, 4, 70, 250, "vg", ""),
    ("hot_sauce", "Hot Sauce", "Pantry", "condiment", 21, 1.3, 4, 0.4, 80, 148, "vg", ""),
    ("maple_syrup", "Maple Syrup", "Pantry", "sweetener", 260, 0, 67, 0.1, 90, 500, "vg", ""),
    ("breadcrumbs", "Breadcrumbs", "Pantry", "bakery", 395, 13, 72, 5, 40, 425, "V", "wheat"),
    ("chips", "Potato Chips", "Snacks", "snack", 536, 7, 53, 35, 80, 200, "vg", ""),
    ("cola", "Cola", "Beverages", "beverage", 42, 0, 10.6, 0, 12, 2000, "vg", ""),
]

BRANDS = {
    "Produce": ["No Name", "PC", "Selection"],
    "Meat": ["No Name", "PC", "Maple Grove"],
    "Seafood": ["No Name", "PC", "Harbour"],
    "Dairy": ["No Name", "PC", "Ridgedale"],
    "Bakery": ["No Name", "PC", "Millbake"],
    "Pantry": ["No Name", "PC", "Casa"],
    "Frozen": ["No Name", "PC", "Northern"],
    "Snacks": ["No Name", "PC", "Crunch"],
    "Beverages": ["No Name", "PC", "Fizz"],
}
MULT = (0.86, 1.0, 1.14)
DEAL = {
    "chicken_thigh", "rice", "broccoli", "eggs", "black_beans", "oats",
    "yogurt_greek", "pasta", "canned_tomato", "lentils", "tuna",
}

# Hand-labelled queries. Clean names are added for every ingredient below.
HARD = [
    ("coriander", "cilantro", "synonym"),
    ("fresh coriander", "cilantro", "synonym"),
    ("spring onions", "green_onion", "synonym"),
    ("scallions", "green_onion", "synonym"),
    ("prawns", "shrimp", "synonym"),
    ("raw prawns", "shrimp", "synonym"),
    ("aubergine", "eggplant", "synonym"),
    ("courgette", "zucchini", "synonym"),
    ("capsicum", "bell_pepper", "synonym"),
    ("red capsicum", "bell_pepper", "synonym"),
    ("rocket", "arugula", "synonym"),
    ("beetroot", "beets", "synonym"),
    ("icing sugar", "powdered_sugar", "synonym"),
    ("bicarbonate of soda", "baking_soda", "synonym"),
    ("bicarb", "baking_soda", "synonym"),
    ("plain flour", "flour", "synonym"),
    ("tinned tomatoes", "canned_tomato", "synonym"),
    ("tinned chopped tomatoes", "canned_tomato", "synonym"),
    ("beef mince", "ground_beef", "synonym"),
    ("minced beef", "ground_beef", "synonym"),
    ("rapini", "broccoli_raab", "synonym"),
    ("greek yoghurt", "yogurt_greek", "synonym"),
    ("plain yoghurt", "yogurt_plain", "synonym"),
    ("yoghurt", "yogurt_plain", "synonym"),
    ("chick peas", "chickpeas", "synonym"),
    ("garbanzo beans", "chickpeas", "synonym"),
    ("rapeseed oil", "canola_oil", "synonym"),
    ("vegetable stock", "stock_veg", "synonym"),
    ("chicken stock", "stock_chicken", "synonym"),
    ("sriracha", "hot_sauce", "synonym"),
    ("brocoli", "broccoli", "typo"),
    ("brocolli", "broccoli", "typo"),
    ("chiken breast", "chicken_breast", "typo"),
    ("chicken thghs", "chicken_thigh", "typo"),
    ("chkn thyghs", "chicken_thigh", "typo"),
    ("zuchini", "zucchini", "typo"),
    ("avacado", "avocado", "typo"),
    ("tumeric", "turmeric", "typo"),
    ("cilanto", "cilantro", "typo"),
    ("siracha", "hot_sauce", "typo"),
    ("boneless skinless chicken breast", "chicken_breast", "paraphrase"),
    ("bone-in chicken thighs", "chicken_thigh", "paraphrase"),
    ("baby spinach", "spinach", "paraphrase"),
    ("firm tofu", "tofu", "paraphrase"),
    ("long grain rice", "rice", "paraphrase"),
    ("canned tuna", "tuna", "paraphrase"),
    ("2% milk", "milk_2", "paraphrase"),
    ("skim milk", "milk_skim", "paraphrase"),
    ("rolled oats", "oats", "paraphrase"),
    ("sweet potatoes", "sweet_potato", "paraphrase"),
    ("green onions", "green_onion", "paraphrase"),
    ("bell peppers", "bell_pepper", "paraphrase"),
    ("ground turkey", "turkey_ground", "paraphrase"),
    ("dijon mustard", "dijon", "paraphrase"),
    ("maple syrup", "maple_syrup", "paraphrase"),
    ("peanut butter", "peanut_butter", "paraphrase"),
    ("olive oil", "olive_oil", "paraphrase"),
    ("black beans canned", "black_beans", "paraphrase"),
    ("cremini mushrooms", "mushroom", "paraphrase"),
    ("romaine lettuce", "lettuce", "paraphrase"),
]

RECIPES = [
    ("sheet_pan_thighs", "Sheet-pan chicken thighs with broccoli and rice",
     "Thighs, broccoli, and rice on one tray.", 40,
     [("chicken_thigh", 380), ("broccoli", 320), ("rice", 160), ("olive_oil", 12), ("garlic", 12), ("salt", 2), ("black_pepper", 1)]),
    ("salmon_tray", "Salmon with sweet potato and spinach",
     "Roasted salmon, sweet potato, and a pan of spinach.", 30,
     [("salmon", 280), ("sweet_potato", 360), ("spinach", 120), ("olive_oil", 12), ("lemon", 40), ("salt", 2)]),
    ("turkey_chili", "Turkey and black bean chili",
     "A pot of ground turkey, black beans, and tomato. No pork.", 35,
     [("turkey_ground", 340), ("black_beans", 250), ("canned_tomato", 300), ("onion", 110), ("cumin", 4), ("chili_flakes", 1), ("salt", 3)]),
    ("shrimp_stirfry", "Shrimp and pepper stir-fry",
     "Shrimp, peppers, and rice with tamari.", 25,
     [("shrimp", 260), ("bell_pepper", 180), ("broccoli", 150), ("rice", 150), ("garlic", 10), ("ginger", 8), ("tamari", 15), ("sesame_oil", 8)]),
    ("chicken_bowl", "Chicken quinoa bowl with tomato and cucumber",
     "Chicken breast, quinoa, and a sharp salad.", 30,
     [("chicken_breast", 320), ("quinoa", 140), ("cucumber", 150), ("tomato", 180), ("feta", 60), ("lemon", 30), ("olive_oil", 12)]),
    ("lentil_pasta", "Lentil tomato pasta",
     "Lentils simmered into a tomato sauce for spaghetti.", 35,
     [("lentils", 160), ("pasta", 170), ("canned_tomato", 280), ("onion", 90), ("carrot", 80), ("garlic", 10), ("olive_oil", 12), ("salt", 2)]),
    ("tofu_stirfry", "Tofu and broccoli stir-fry",
     "Firm tofu, broccoli, and rice.", 25,
     [("tofu", 350), ("bell_pepper", 160), ("broccoli", 180), ("rice", 150), ("garlic", 10), ("ginger", 8), ("tamari", 15), ("sesame_oil", 6)]),
    ("tuna_chickpea", "Tuna and chickpea salad",
     "Canned tuna, chickpeas, spinach, and lemon.", 15,
     [("tuna", 300), ("chickpeas", 240), ("spinach", 100), ("lemon", 30), ("olive_oil", 12), ("parsley", 8), ("onion", 50)]),
    ("egg_skillet", "Egg and potato skillet",
     "Eggs, potato, spinach, and a little cheddar.", 25,
     [("eggs", 220), ("potato", 320), ("spinach", 80), ("cheddar", 40), ("olive_oil", 10), ("salt", 2)]),
    ("chickpea_curry", "Chickpea coconut curry",
     "Chickpeas in tomato and coconut milk over rice.", 30,
     [("chickpeas", 400), ("canned_tomato", 200), ("onion", 100), ("garlic", 10), ("ginger", 10), ("curry_powder", 6), ("coconut_milk", 160), ("rice", 160)]),
    ("beef_stirfry", "Beef and broccoli stir-fry",
     "Sirloin, broccoli, and rice.", 25,
     [("sirloin", 300), ("bell_pepper", 150), ("broccoli", 180), ("rice", 150), ("tamari", 15), ("garlic", 10), ("ginger", 8)]),
    ("cod_potatoes", "Cod with potatoes and peas",
     "Baked cod, potatoes, and peas.", 35,
     [("cod", 320), ("potato", 400), ("peas", 140), ("lemon", 30), ("olive_oil", 12), ("parsley", 8), ("salt", 2)]),
    ("bean_tacos", "Black bean tacos",
     "Beans, tortillas, lettuce, and salsa.", 20,
     [("black_beans", 300), ("tortilla", 140), ("lettuce", 80), ("cheddar", 40), ("salsa", 60), ("onion", 50), ("cumin", 2)]),
    ("pork_chops", "Pork chops with apples and potatoes",
     "Pan pork chops, apple, and potato.", 30,
     [("pork_chop", 360), ("apple", 180), ("potato", 340), ("olive_oil", 10), ("salt", 2), ("black_pepper", 1)]),
    ("sausage_pasta", "Pork sausage pasta",
     "Italian pork sausage and tomato on spaghetti.", 30,
     [("italian_sausage", 280), ("pasta", 180), ("canned_tomato", 260), ("onion", 80), ("garlic", 10), ("chili_flakes", 1)]),
    ("steak_salad", "Sirloin salad with arugula",
     "Sliced sirloin over romaine, tomato, and arugula.", 20,
     [("sirloin", 280), ("lettuce", 140), ("arugula", 40), ("tomato", 120), ("cucumber", 80), ("feta", 40), ("olive_oil", 12), ("lemon", 20)]),
    ("miso_cod", "Miso cod with spinach",
     "Cod brushed with miso, rice, and spinach.", 25,
     [("cod", 300), ("miso", 24), ("rice", 150), ("spinach", 140), ("green_onion", 30), ("sesame_oil", 6)]),
    ("mushroom_eggs", "Mushroom and spinach eggs",
     "Eggs with cremini mushrooms and cheddar.", 15,
     [("eggs", 200), ("mushroom", 140), ("cheddar", 40), ("spinach", 60), ("olive_oil", 8)]),
    ("bean_bowl", "Black bean rice bowl",
     "Beans, rice, corn, and salsa.", 20,
     [("black_beans", 260), ("rice", 160), ("corn", 120), ("salsa", 60), ("lettuce", 70), ("lime", 30)]),
    ("turkey_pasta", "Turkey meat sauce",
     "Ground turkey tomato sauce on spaghetti.", 30,
     [("turkey_ground", 320), ("breadcrumbs", 20), ("canned_tomato", 240), ("pasta", 170), ("parmesan", 20), ("garlic", 8), ("onion", 70)]),
    ("salmon_quinoa", "Salmon quinoa salad",
     "Salmon, quinoa, cucumber, and arugula.", 25,
     [("salmon", 260), ("quinoa", 130), ("cucumber", 120), ("arugula", 50), ("lemon", 20), ("olive_oil", 10)]),
    ("lentil_soup", "Lentil vegetable soup",
     "Lentils, carrot, celery, and vegetable broth.", 35,
     [("lentils", 180), ("carrot", 120), ("onion", 90), ("celery", 70), ("canned_tomato", 120), ("stock_veg", 300), ("cumin", 3), ("salt", 2)]),
    ("chicken_soup", "Chicken and rice soup",
     "Chicken breast, rice, and chicken broth.", 35,
     [("chicken_breast", 280), ("carrot", 80), ("celery", 60), ("onion", 70), ("stock_chicken", 400), ("rice", 90)]),
    ("pork_rice", "Pork and egg fried rice",
     "Ground pork, egg, and rice.", 20,
     [("ground_pork", 220), ("rice", 200), ("eggs", 100), ("green_onion", 30), ("tamari", 15), ("peas", 80)]),
    ("lentil_salad", "Lentil salad with tomato and cucumber",
     "Lentils, tomato, cucumber, and feta.", 15,
     [("lentils", 180), ("tomato", 160), ("cucumber", 120), ("feta", 50), ("olive_oil", 12), ("onion", 40), ("parsley", 8)]),
    ("shrimp_tacos", "Shrimp tacos",
     "Shrimp, cabbage, tortillas, and lime.", 20,
     [("shrimp", 240), ("tortilla", 120), ("cabbage", 90), ("lime", 30), ("yogurt_greek", 50), ("cilantro", 10)]),
    ("tofu_scramble", "Tofu scramble with spinach",
     "Tofu, turmeric, spinach, and onion.", 20,
     [("tofu", 320), ("turmeric", 3), ("spinach", 90), ("onion", 60), ("olive_oil", 10), ("salt", 2)]),
    ("beef_chili", "Beef and kidney bean chili",
     "Ground beef, kidney beans, and tomato.", 40,
     [("ground_beef", 320), ("kidney_beans", 240), ("canned_tomato", 280), ("onion", 90), ("chili_flakes", 2), ("cumin", 4), ("salt", 2)]),
    ("turkey_bowl", "Turkey rice bowl",
     "Ground turkey, rice, and peppers.", 25,
     [("turkey_ground", 300), ("rice", 160), ("bell_pepper", 160), ("onion", 70), ("olive_oil", 10), ("paprika", 2), ("salt", 2)]),
    ("egg_fried_rice", "Egg fried rice with peas",
     "Eggs, rice, peas, and green onion.", 20,
     [("eggs", 200), ("rice", 180), ("peas", 100), ("green_onion", 30), ("tamari", 12), ("sesame_oil", 6)]),
]


def decode_tags(tag: str) -> dict:
    vegan = "v" in tag
    vegetarian = vegan or "V" in tag
    flags = {
        "vegan": vegan,
        "vegetarian": vegetarian,
        "gluten_free": "g" in tag,
        "pork": "P" in tag,
        "beef": "B" in tag,
        "dairy": "D" in tag,
        "nuts": "N" in tag,
        "shellfish": "S" in tag,
        "soy": "Y" in tag,
        "fish": "F" in tag,
        "egg": "E" in tag,
    }
    if flags["pork"] or flags["beef"] or flags["fish"] or flags["shellfish"]:
        if flags["vegetarian"] or flags["vegan"]:
            raise SystemExit(f"animal product tagged vegetarian: {tag}")
    if (flags["dairy"] or flags["egg"]) and flags["vegan"]:
        raise SystemExit(f"vegan item has dairy or egg: {tag}")
    return flags


def brand_code(brand: str) -> str:
    letters = "".join(ch for ch in brand.upper() if ch.isalnum())
    return letters[:6] or "SKU"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ingredients = []
    products = []
    by_id = {}
    for row in ROWS:
        iid, name, aisle, group, kcal, protein, carbs, fat, cents, pack, tag, allergens = row
        flags = decode_tags(tag)
        allergen_list = [a for a in allergens.split(",") if a]
        ing = {
            "id": iid,
            "name": name,
            "aisle": aisle,
            "group": group,
            "kcal_100": kcal,
            "protein_100": protein,
            "carbs_100": carbs,
            "fat_100": fat,
            "allergens": allergen_list,
            **flags,
        }
        ingredients.append(ing)
        by_id[iid] = ing
        brands = BRANDS[aisle]
        sizes = [pack, int(round(pack * 1.35)), max(40, int(round(pack * 0.72)))]
        for idx, brand in enumerate(brands):
            size = sizes[idx]
            price = max(99, int(round(cents * size / 100 * MULT[idx])))
            stock = 10 + idx * 6
            if iid in {"feta", "lentils"}:
                stock = 0
            if iid == "chicken_thigh" and idx == 0:
                stock = 0
            if iid == "salmon" and idx == 2:
                stock = 0
            deal = 0.25 if idx == 1 and iid in DEAL else 0.0
            sku = f"{brand_code(brand)}-{iid.replace('_', '-').upper()}-{idx + 1}"
            products.append({
                "sku": sku,
                "ingredient_id": iid,
                "ingredient": name,
                "name": name,
                "brand": brand,
                "aisle": aisle,
                "group": group,
                "size_g": size,
                "price_cents": price,
                "stock": stock,
                "deal_pct": deal,
                "kcal_100": kcal,
                "protein_100": protein,
                "carbs_100": carbs,
                "fat_100": fat,
                "allergens": allergen_list,
                "search_text": f"{brand} {name}",
                **flags,
            })

    recipes = []
    for rid, name, blurb, minutes, items in RECIPES:
        for iid, grams in items:
            if iid not in by_id:
                raise SystemExit(f"recipe {rid} uses unknown ingredient {iid}")
            if grams <= 0:
                raise SystemExit(f"bad grams in {rid}")
        protein = sum(by_id[iid]["protein_100"] * grams / 100 for iid, grams in items)
        tags = []
        if all(by_id[iid]["vegan"] for iid, _ in items):
            tags.append("vegan")
        if all(by_id[iid]["vegetarian"] for iid, _ in items):
            tags.append("vegetarian")
        if all(by_id[iid]["gluten_free"] for iid, _ in items):
            tags.append("gluten-free")
        if not any(by_id[iid]["dairy"] for iid, _ in items):
            tags.append("dairy-free")
        if not any(by_id[iid]["pork"] for iid, _ in items):
            tags.append("pork-free")
        if protein / 2 >= 30:
            tags.append("high-protein")
        recipes.append({
            "id": rid,
            "name": name,
            "blurb": blurb,
            "minutes": minutes,
            "tags": tags,
            "protein_g": round(protein, 1),
            "ingredients": [{"ingredient_id": iid, "grams": grams} for iid, grams in items],
        })

    skus_by_ing: dict[str, list[dict]] = {}
    for product in products:
        skus_by_ing.setdefault(product["ingredient_id"], []).append(product)

    rng = np.random.default_rng(7)
    groups: dict[str, list[str]] = {}
    for ing in ingredients:
        groups.setdefault(ing["group"], []).append(ing["id"])

    def pick_sku(iid: str) -> str:
        choices = skus_by_ing[iid]
        return choices[int(rng.integers(0, len(choices)))]["sku"]

    orders = []
    n_orders = 6500
    for i in range(n_orders):
        recipe = recipes[int(rng.integers(0, len(recipes)))]
        skus = []
        for item in recipe["ingredients"]:
            iid = item["ingredient_id"]
            group = by_id[iid]["group"]
            siblings = [s for s in groups[group] if s != iid]
            if siblings and rng.random() < 0.38:
                iid = siblings[int(rng.integers(0, len(siblings)))]
            skus.append(pick_sku(iid))
        if rng.random() < 0.48:
            skus.append(pick_sku("banana"))
        if rng.random() < 0.36:
            skus.append(pick_sku("milk_2"))
        if rng.random() < 0.24:
            skus.append(pick_sku("bacon"))
        if rng.random() < 0.18:
            skus.append(pick_sku("bread"))
        if rng.random() < 0.16:
            skus.append(pick_sku("chips"))
        if rng.random() < 0.12:
            skus.append(pick_sku("cola"))
        orders.append({
            "order_id": f"B{i:05d}",
            "split": "test" if i % 5 == 0 else "train",
            "skus": skus,
        })

    labels = []
    seen = set()

    def add(query: str, iid: str, kind: str) -> None:
        q = " ".join(query.lower().split())
        if q in seen:
            return
        if iid not in by_id:
            raise SystemExit(f"label {query!r} uses unknown {iid}")
        seen.add(q)
        labels.append({
            "query": q,
            "ingredient_id": iid,
            "relevant_skus": [p["sku"] for p in skus_by_ing[iid]],
            "kind": kind,
        })

    for query, iid, kind in HARD:
        add(query, iid, kind)
    for ing in ingredients:
        add(ing["name"], ing["id"], "clean")
    for ing in ingredients:
        if len(labels) >= 200:
            break
        add(f"{ing['name']} family pack", ing["id"], "paraphrase")
    for ing in ingredients:
        if len(labels) >= 200:
            break
        add(f"buy {ing['name'].lower()}", ing["id"], "paraphrase")
    if len(labels) < 200:
        raise SystemExit(f"only {len(labels)} labels")
    labels = labels[:200]
    kinds: dict[str, int] = {}
    for label in labels:
        kinds[label["kind"]] = kinds.get(label["kind"], 0) + 1

    write_jsonl(OUT / "ingredients.jsonl", ingredients)
    write_jsonl(OUT / "products.jsonl", products)
    write_jsonl(OUT / "recipes.jsonl", recipes)
    write_jsonl(OUT / "orders.jsonl", orders)
    write_jsonl(OUT / "labels.jsonl", labels)
    manifest = {
        "ingredients": len(ingredients),
        "products": len(products),
        "recipes": len(recipes),
        "orders": len(orders),
        "train_orders": sum(1 for o in orders if o["split"] == "train"),
        "test_orders": sum(1 for o in orders if o["split"] == "test"),
        "labels": len(labels),
        "label_kinds": kinds,
        "notes": "Prices, stock, and flyer deals are synthetic. Baskets follow the Instacart order schema.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
