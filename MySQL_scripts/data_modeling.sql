SET GLOBAL local_infile = TRUE;
SET SQL_SAFE_UPDATES = 0;

DROP DATABASE IF EXISTS `transaction`;
CREATE DATABASE IF NOT EXISTS `transaction`;

USE `transaction`;

DROP TABLE IF EXISTS `transaction`.`address`;
CREATE TABLE IF NOT EXISTS `transaction`.`address` (
	`customer_id` INT(8) PRIMARY KEY NOT NULL,
    `street` VARCHAR(50) NOT NULL,
    `state` VARCHAR(20) NOT NULL,
    `country` VARCHAR(30) NOT NULL,
    `post_code` INT(8) NOT NULL
)DEFAULT CHARACTER SET=UTF8;

LOAD DATA LOCAL INFILE '/Users/chonge/Documents/Document/University_of_Chicago/transaction_project/raw_data/address.csv' 
INTO TABLE address 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 2 ROWS
(@customer_id,@address,@postcode,@state,@country) 
set customer_id=@customer_id,street=@address,post_code=@postcode,state=@state,country=@country ;

UPDATE address
SET state = 'NSW'
WHERE state = 'New South Wales';

UPDATE address
SET state = 'VIC'
WHERE state = 'Victoria';

DROP TABLE IF EXISTS `transaction`.`demographics`;
CREATE TABLE IF NOT EXISTS `transaction`.`demographics` (
	`customer_id` INT(8) PRIMARY KEY NOT NULL,
    `first_name` VARCHAR(50) NOT NULL,
	`last_name` VARCHAR(50) NOT NULL,
    `gender` VARCHAR(20),
    `num_prev_purchase` VARCHAR(30),
    `DOB` DATE,
    `age` INT(5),
    `job_title` VARCHAR(80),
    `job_industry` VARCHAR(50),
    `wealth_segment` VARCHAR(50),
    `owns_car` VARCHAR(10)
)DEFAULT CHARACTER SET=UTF8;

LOAD DATA LOCAL INFILE '/Users/chonge/Documents/Document/University_of_Chicago/transaction_project/raw_data/customer.csv' 
INTO TABLE demographics 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@customer_id,@name,@gender,@past_3_years_bike_related_purchases,@DOB,@age,@job_title,@job_industry_category,@wealth_segment,@owns_car) 
set customer_id=@customer_id,first_name=TRIM(SUBSTRING_INDEX(@name, ' ', 1)),last_name=TRIM(SUBSTRING_INDEX(@name, ' ', -1)),gender=@gender,num_prev_purchase=@past_3_years_bike_related_purchases,DOB=STR_TO_DATE(@DOB,'%Y-%c-%d'),age=@age,job_title=@job_title, job_industry=@job_industry_category,wealth_segment=@wealth_segment,owns_car=@owns_car;

DROP TABLE IF EXISTS `transaction`.`dim_customer`;
CREATE TABLE IF NOT EXISTS `transaction`.`dim_customer`
SELECT * FROM demographics FULL JOIN address USING(customer_id);

ALTER TABLE `transaction`.`dim_customer`
ADD PRIMARY KEY (customer_id);

DROP TABLE IF EXISTS `transaction`.`dim_product`;
CREATE TABLE IF NOT EXISTS `transaction`.`dim_product` (
    `product_id`  INT(8) PRIMARY KEY NOT NULL,
    `brand` VARCHAR(50),
    `product_line` VARCHAR(50),
    `product_class` VARCHAR(30),
    `product_size` VARCHAR(30),
    `list_price` DECIMAL(10,2),
    `standard_cost` DECIMAL(10,2)
)DEFAULT CHARACTER SET=UTF8;

LOAD DATA LOCAL INFILE '/Users/chonge/Documents/Document/University_of_Chicago/transaction_project/raw_data/transaction.csv' 
INTO TABLE dim_product 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 2 ROWS
(@transaction_id,@product_id,@customer_id,@transaction_date,@online_order,@order_status,@brand,@product_line,@product_class,@product_size,@list_price,@standard_cost) 
set product_id=@product_id,brand=@brand,product_line=@product_line,product_class=@product_class,product_size=@product_size,list_price=@list_price,standard_cost=replace(@standard_cost, '$', '');

DROP TABLE IF EXISTS `transaction`.`transaction`;
CREATE TABLE IF NOT EXISTS `transaction`.`transaction` (
	`transaction_id` INT(8) PRIMARY KEY NOT NULL,
    `product_id` INT(8) NOT NULL,
    `customer_id` INT(8) NOT NULL,
    `transaction_date` DATE,
    `online_order` VARCHAR(10),
    `order_status`VARCHAR(20),
	FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
	FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
)DEFAULT CHARACTER SET=UTF8;

LOAD DATA LOCAL INFILE '/Users/chonge/Documents/Document/University_of_Chicago/transaction_project/raw_data/transaction.csv' 
INTO TABLE transaction 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 2 ROWS
(@transaction_id,@product_id,@customer_id,@transaction_date,@online_order,@order_status) 
set transaction_id=@transaction_id,product_id=@product_id,customer_id=@customer_id,transaction_date=STR_TO_DATE(@transaction_date, '%d/%m/%Y'),online_order=@online_order,order_status=@order_status;

CREATE TABLE `transaction`.`dim_time` AS 
SELECT transaction_id, transaction_date,
DAYNAME(transaction_date) AS day_of_week,
EXTRACT(YEAR FROM transaction_date) AS year,
EXTRACT(MONTH FROM transaction_date) AS month,
EXTRACT(DAY FROM transaction_date) AS day
FROM transaction;

ALTER TABLE `transaction`.`dim_time`
ADD PRIMARY KEY(transaction_id);

CREATE TABLE `transaction`.`dim_order` AS 
SELECT transaction_id, online_order, order_status
FROM transaction;

ALTER TABLE `transaction`.`dim_order`
ADD PRIMARY KEY(transaction_id);

ALTER TABLE `transaction`.`transaction`
DROP COLUMN transaction_date,
DROP COLUMN online_order,
DROP COLUMN order_status,
ADD FOREIGN KEY (transaction_id) REFERENCES dim_time(transaction_id),
ADD FOREIGN KEY (transaction_id) REFERENCES dim_order(transaction_id);

CREATE VIEW customer_profile AS
SELECT *
FROM dim_customer LEFT JOIN 
(SELECT customer_id, COUNT(*) AS num_purchase, SUM(list_price) AS total_purchase
FROM transaction LEFT JOIN 
(SELECT product_id, list_price FROM dim_product) AS prod
USING(product_id)
GROUP BY customer_id) AS trans
USING(customer_id)
ORDER BY customer_id;

CREATE VIEW product_profile AS
SELECT *, 
dim_product.list_price-dim_product.standard_cost AS net_profit,
(dim_product.list_price-dim_product.standard_cost)*trans.num_purchase AS total_profit
FROM dim_product LEFT JOIN 
(SELECT product_id, COUNT(*) AS num_purchase
FROM transaction
GROUP BY product_id) AS trans 
USING(product_id)
ORDER BY product_id;

CREATE VIEW order_profile AS
SELECT *
FROM transaction LEFT JOIN dim_time USING(transaction_id) 
LEFT JOIN dim_order USING(transaction_id) ;

