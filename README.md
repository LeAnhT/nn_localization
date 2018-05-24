# nn_localization

In this little project we want to figure out, if it is possible to locate a target-point with just
the distances to the meassurement devices and the location of the meassurement devices given.

A common approach would be solving this problem by using "trilateration".
This approach on the other side doesnt work well when the meassured values are inaccurate.
So we are trying to solve this with a simple feed-forward Neural Network.

The project consists of 4 python files.

meassurement_tools.py: contains utility-methods to meassure the distance between a target-point and the meassurement devices.
plot_helper.py contains some utility-methods for plotting the results
data_creator.py contains methods for generating our training - and testdata
nn_trainer.py contains the code, where we build, train and evaluate the neural net.
