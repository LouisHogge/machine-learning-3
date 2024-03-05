# Machine Learning Project 3: Competition

## Project Description
The aim of this project is to get some experience with designing a solution to a more complex problem on large real data, using the different tools seen in the course.

## Competition Description
Over time, taxi technologies have evolved, notably the use of electronic dispatch systems. With these systems, taxis provide GPS information about their location, but usually do not provide their destination. Therefore, it might be hard to know which taxi to contact when planning a future trip. We would like to tackle this issue, so that we can know which taxis will end near a new requested pick up. This would highly improve the efficiency of taxi systems.

More precisely, the goal here is to design a solution that predicts the destination of a taxi trip given its partial initial trajectory, which is of varying length. For that, you have access to a database of full trajectories, which you can use to train your solution.

Along with the trajectories which can certainly help predicting the destination, you have access to some metadata, such as the taxi ID, the phone ID of the customer when the taxi was ordered with a call, and more. These pieces of information could also perhaps help in narrowing down the destination based on the historical training data.

For the predictions, the information the model has access to is in the same format, except the trajectory is now only a cut-off initial trajectory of varying length.

## Dataset Description
The dataset comprises a full year of trajectories for all the 442 taxis in Porto, Portugal. This results in a table of 1,710,670 rows.

Each ride is categorized into 3 categories:
- **Taxi central-based**: Trips dispatched by the central dispatch unit.
- **Stand-based**: Trips directly requested at a taxi stand.
- **Non-taxi central-based**: Trips requested directly from a random street.

The dataset includes the following nine features:
- `TRIP_ID`: Identifier for each trip (some IDs are not unique).
- `CALL_TYPE`: Category of the trip (A, B, or C).
- `ORIGIN_CALL`: Unique identifier for the phone number used to order the trip (only if `CALL_TYPE` is A).
- `ORIGIN_STAND`: Unique identifier for the taxi stand (only if `CALL_TYPE` is B).
- `TAXI_ID`: Unique identifier for the taxi driver.
- `TIMESTAMP`: Unix timestamp of the trip start.
- `DAY_TYPE`: Identifies the type of day for the trip (Normal days, Holidays, Days before holidays).
- `MISSING_DATA`: Indicates if the GPS stream is incomplete (True/False).
- `POLYLINE`: List of GPS coordinates for each 15 seconds of the trip, represented as a string.

The dataset is split into training and test sets, with the training data available in `train.csv` and the test data in `test.csv`. The test data format mirrors the training data, with the exception of the `POLYLINE` feature, which only contains partial initial trajectories for the purpose of prediction.

## How to Use the Project
Use *competition.ipynb*.
