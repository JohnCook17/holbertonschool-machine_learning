-- creates the table users with id email name and country

DROP TABLE IF EXISTS `users`;

CREATE TABLE `users`(
    `id` INT(11) NOT NULL AUTO_INCREMENT,
    `email` CHAR(255) NOT NULL UNIQUE,
    `name` CHAR(255),
    `country` ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US', 
    PRIMARY KEY (`id`)
);