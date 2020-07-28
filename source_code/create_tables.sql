CREATE TABLE IF NOT EXISTS airlines (
    iata_code varchar(2) NOT NULL,
    airline varchar(30) NOT NULL,
    PRIMARY KEY (IATA_CODE)
);

COPY airlines FROM '/Users/binhhoang/Google Drive/Data Science/'
'Metis Bootcamp/Projects/project_3/flight_classification/raw_data/airlines.csv'
DELIMITER ',' CSV HEADER;

CREATE TABLE IF NOT EXISTS airports (
    iata_code varchar(3) NOT NULL,
    airport varchar(100) NOT NULL,
    city varchar(50) NOT NULL,
    state varchar(2) NOT NULL,
    country varchar(3) NOT NULL,
    latitude real DEFAULT NULL,
    longitude real DEFAULT NULL,
    PRIMARY KEY (iata_code)
);

COPY airports FROM '/Users/binhhoang/Google Drive/Data Science/'
'Metis Bootcamp/Projects/project_3/flight_classification/raw_data/airports.csv'
DELIMITER ',' CSV HEADER;

CREATE TABLE IF NOT EXISTS flights (
    year int NOT NULL,
    month int NOT NULL,
    day int NOT NULL,
    day_of_week int NOT NULL,
    airline varchar(2) NOT NULL,
    flight_number int NOT NULL,
    tail_number varchar(6) DEFAULT NULL,
    origin_airport varchar(5) NOT NULL,
    destination_airport varchar(5) NOT NULL,
    scheduled_departure varchar(4) NOT NULL,
    departure_time varchar(4) DEFAULT NULL,
    departure_delay varchar(5) DEFAULT NULL,
    taxi_out varchar(5) DEFAULT NULL,
    wheels_off varchar(4) DEFAULT NULL,
    scheduled_time varchar(5) DEFAULT NULL,
    elapsed_time varchar(5) DEFAULT NULL,
    air_time varchar(5) DEFAULT NULL,
    distance varchar(5) DEFAULT NULL,
    wheels_on varchar(4) DEFAULT NULL,
    taxi_in varchar(5) DEFAULT NULL,
    scheduled_arrival varchar(4) NOT NULL,
    arrival_time varchar(4) DEFAULT NULL,
    arrival_delay varchar(5) DEFAULT NULL,
    diverted int NOT NULL,
    cancelled int NOT NULL,
    cancellation_reason varchar(5) DEFAULT NULL,
    air_system_delay varchar(5) DEFAULT NULL,
    security_delay varchar(5) DEFAULT NULL,
    airline_delay varchar(5) DEFAULT NULL,
    late_aircraft varchar(5) DEFAULT NULL,
    weather_delay varchar(5) DEFAULT NULL
);

COPY flights FROM '/Users/binhhoang/Google Drive/Data Science/'
'Metis Bootcamp/Projects/project_3/flight_classification/raw_data/flights.csv'
DELIMITER ',' CSV HEADER;
