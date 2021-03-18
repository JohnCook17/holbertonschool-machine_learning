-- Calculates a new correction for a student
DELIMITER $$
DROP PROCEDURE IF EXISTS AddBonus;
CREATE PROCEDURE AddBonus(
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN newscore INT)
BEGIN
    IF NOT EXISTS(
        SELECT *
            FROM projects
                WHERE projects.name = project_name) THEN
                    INSERT INTO projects(name) VALUES(project_name);
    END IF;


END;
$$
DELIMITER ;
