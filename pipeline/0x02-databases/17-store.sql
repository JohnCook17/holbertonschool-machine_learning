-- triger that monitors stock of items, can go negative
DROP TRIGGER IF EXISTS item_update;
DELIMITER $$
CREATE TRIGGER item_update
AFTER INSERT
ON orders 
BEGIN
    insert into items(name, quantity)
    SELECT items.name, items.quantity - IFNULL(orders.number, 0) 
    FROM items LEFT JOIN orders ON items.name = orders.item_name
    ORDER BY items.name;
END;
$$
DELIMITER ;
