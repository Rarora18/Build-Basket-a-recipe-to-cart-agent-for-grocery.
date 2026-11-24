CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    ingredient_id TEXT NOT NULL,
    ingredient TEXT NOT NULL,
    name TEXT NOT NULL,
    brand TEXT NOT NULL,
    aisle TEXT NOT NULL,
    grp TEXT NOT NULL,
    size_g INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    stock INTEGER NOT NULL,
    deal_pct DOUBLE PRECISION NOT NULL,
    kcal_100 DOUBLE PRECISION NOT NULL,
    protein_100 DOUBLE PRECISION NOT NULL,
    carbs_100 DOUBLE PRECISION NOT NULL,
    fat_100 DOUBLE PRECISION NOT NULL,
    allergens TEXT[] NOT NULL DEFAULT '{}',
    pork BOOLEAN NOT NULL,
    beef BOOLEAN NOT NULL,
    dairy BOOLEAN NOT NULL,
    nuts BOOLEAN NOT NULL,
    shellfish BOOLEAN NOT NULL,
    soy BOOLEAN NOT NULL,
    fish BOOLEAN NOT NULL,
    egg BOOLEAN NOT NULL,
    vegetarian BOOLEAN NOT NULL,
    vegan BOOLEAN NOT NULL,
    gluten_free BOOLEAN NOT NULL,
    search_text TEXT NOT NULL,
    embedding vector(128)
);

CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    blurb TEXT NOT NULL,
    minutes INTEGER NOT NULL,
    tags TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
    recipe_id TEXT NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    ingredient_id TEXT NOT NULL,
    grams INTEGER NOT NULL,
    PRIMARY KEY (recipe_id, ingredient_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    split TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    sku TEXT NOT NULL,
    PRIMARY KEY (order_id, position)
);

CREATE INDEX IF NOT EXISTS products_ingredient_idx ON products (ingredient_id);
CREATE INDEX IF NOT EXISTS products_aisle_idx ON products (aisle);
CREATE INDEX IF NOT EXISTS products_search_idx ON products USING GIN (to_tsvector('english', search_text));
