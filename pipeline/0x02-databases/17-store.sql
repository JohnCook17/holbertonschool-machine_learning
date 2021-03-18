-- triger that monitors stock of items, can go negative
DROP TRIGGER IF EXISTS item_update;
DROP TABLE IF EXISTS my_temp;
CREATE TABLE my_temp(id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), number int);
DELIMITER $$
CREATE TRIGGER item_update
AFTER INSERT
ON orders FOR EACH ROW
BEGIN
    IF NOT EXISTS(SELECT id FROM my_temp) THEN
        INSERT INTO my_temp(name, number)
        SELECT item_name, number FROM orders;
        INSERT INTO items(quantity)
        SELECT items.quantity - IFNULL(orders.number, 0) 
        FROM items LEFT JOIN orders ON items.name = orders.item_name
        WHERE items.name = orders.item_name
        ORDER BY items.name;
    END IF;
END;
$$
DELIMITER ;
