-- Always start with small queries, check if they work, build up,
-- and check with each incremental step.

SELECT
    f.*,
    a.airline AS airline_name
FROM flights AS f
    LEFT JOIN airlines AS a
        ON f.airline = a.iata_code
    LIMIT 5;

SELECT
    f.*,
    a.airline AS airline_name,
    o.airport AS origin_airport_name,
    o.city AS origin_airport_city,
    o.state AS origin_airport_state,
    o.latitude AS destination_airport_latitude,
    o.longitude AS destination_airport_longitude
FROM flights AS f
    LEFT JOIN airlines AS a
        ON f.airline = a.iata_code
    LEFT JOIN airports AS o
        ON f.origin_airport = o.iata_code
    LIMIT 5;

SELECT
    f.*,
    a.airline AS airline_name,
    o.airport AS origin_airport_name,
    o.city AS origin_airport_city,
    o.state AS origin_airport_state,
    o.latitude AS origin_airport_latitude,
    o.longitude AS origin_airport_longitude,
    d.airport AS destination_airport_name,
    d.city AS destination_airport_city,
    d.state AS destination_airport_state,
    d.latitude AS destination_airport_latitude,
    d.longitude AS destination_airport_longitude
FROM flights AS f
    LEFT JOIN airlines AS a
        ON f.airline = a.iata_code
    LEFT JOIN airports AS o
        ON f.origin_airport = o.iata_code
    LEFT JOIN airports AS d
        ON f.destination_airport = d.iata_code
    LIMIT 5;

CREATE VIEW combined AS (
    SELECT
        f.*,
        a.airline AS airline_name,
        o.airport AS origin_airport_name,
        o.city AS origin_airport_city,
        o.state AS origin_airport_state,
        o.latitude AS origin_airport_latitude,
        o.longitude AS origin_airport_longitude,
        d.airport AS destination_airport_name,
        d.city AS destination_airport_city,
        d.state AS destination_airport_state,
        d.latitude AS destination_airport_latitude,
        d.longitude AS destination_airport_longitude
    FROM flights AS f
        LEFT JOIN airlines AS a
            ON f.airline = a.iata_code
        LEFT JOIN airports AS o
            ON f.origin_airport = o.iata_code
        LEFT JOIN airports AS d
            ON f.destination_airport = d.iata_code
);

CREATE VIEW sample AS (
    SELECT
        f.*,
        a.airline AS airline_name,
        o.airport AS origin_airport_name,
        o.city AS origin_airport_city,
        o.state AS origin_airport_state,
        o.latitude AS origin_airport_latitude,
        o.longitude AS origin_airport_longitude,
        d.airport AS destination_airport_name,
        d.city AS destination_airport_city,
        d.state AS destination_airport_state,
        d.latitude AS destination_airport_latitude,
        d.longitude AS destination_airport_longitude
    FROM flights AS f
        LEFT JOIN airlines AS a
            ON f.airline = a.iata_code
        LEFT JOIN airports AS o
            ON f.origin_airport = o.iata_code
        LEFT JOIN airports AS d
            ON f.destination_airport = d.iata_code
    ORDER BY RANDOM()
    LIMIT 100000
);
