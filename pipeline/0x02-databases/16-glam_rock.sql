-- list all glam rock bands by longevity

SELECT `band_name`, IFNULL(`split`, YEAR(CURDATE())) - `formed` AS `lifespan` FROM metal_bands WHERE `style` LIKE '%Glam rock%' ORDER BY `lifespan` DESC;