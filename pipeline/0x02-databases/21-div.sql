DELIMITER $$
DROP FUNCTION IF EXISTS SafeDiv;
CREATE FUNCTION SafeDiv (f_1 FLOAT, f_2 FLOAT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE answer FLOAT;

    IF f_2 = 0 THEN
        SET answer = 0;
    ELSEIF IFNULL(f_2, 0) THEN
        SET answer = 0;
    ELSE
        SET answer = f_1 / f_2;
    END IF;

    RETURN(answer);
END;
$$
DELIMITER ;