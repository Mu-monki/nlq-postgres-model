CREATE DATABASE research_db;
USE research_db;

CREATE TABLE studies (
    id INT PRIMARY KEY,
    title VARCHAR(100),
    author VARCHAR(50),
    year INT,
    citations INT,
    domain ENUM('CRISPR', 'Neuroscience', 'Cancer')
);

CREATE TABLE experiments (
    id INT PRIMARY KEY,
    study_id INT,
    method VARCHAR(50),
    p_value FLOAT,
    effect_size FLOAT,
    FOREIGN KEY (study_id) REFERENCES studies(id)
);

INSERT INTO studies VALUES
(1, 'CRISPR-Cas9 Gene Editing', 'Zhang et al', 2021, 450, 'CRISPR'),
(2, 'Alzheimers Mouse Model', 'Chen et al', 2022, 210, 'Neuroscience'),
(3, 'Immunotherapy Response', 'Wilson et al', 2023, 890, 'Cancer');

INSERT INTO experiments VALUES
(101, 1, 'RCT', 0.03, 1.2),
(102, 1, 'Meta-Analysis', 0.001, 0.8),
(103, 2, 'Observational', 0.12, 0.4),
(104, 3, 'Double-Blind', 0.008, 2.1);