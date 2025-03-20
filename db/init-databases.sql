-- Création des tables
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    image_name TEXT NOT NULL,
    predicted_result TEXT NOT NULL,
    user_feedback TEXT NOT NULL,
    is_good BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE feedback_counter (
    id SERIAL PRIMARY KEY,
    negative_count INTEGER DEFAULT 0
);

-- Initialise avec une ligne unique
INSERT INTO feedback_counter (negative_count) VALUES (0);
