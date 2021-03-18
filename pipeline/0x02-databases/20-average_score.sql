-- stored procedure that computes average given a student id
DELIMITER $$
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    UPDATE users
    SET users.average_score = AVG(corrections.score) WHERE users.id = user_id;

END;
$$
DELIMITER ;
