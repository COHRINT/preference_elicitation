#from julia.api import Julia
# jl = Julia()
from julia import Main
# import julia as jl
# jl.install()
import sys
import numpy as np

sys.path.insert(0, './python_utils/')
import python_utils.sample_image as image_sampling


# import rospy


class processing_type:
    RANDOM = 1
    USER = 2


class PreferenceAlgorithm:
    def __init__(self):
        # Constant definition
        self.num_particles = 500  # Number of particles generated in the particle filter. Final value should ~= 4000
        self.num_samples = 10  # The number of random samples over the image
        self.p_sample = 10  # Number of points that the algorithm iterates over
        self.num_guess = 5  # Number of guesses performed by the algorithm before final propagation
        self.init_guess = 5  # Number of points before the algorithm starts running
        self.num_final_guess = 10  # Final number of points that will be selected by the algorithm
        self.user_points = []  # Points selected by the user
        self.user_points_msgs = []  # Messages of user points received
        self.POMDP = []  # Current POMDP instantiation
        self.belief = []  # Belief structure
        self.phi = []  # Current best user estimate
        self.best_points_idx = []
        self.best_points_phi = []
        self.guess_points = []  # Points that can be suggested by the algorithm
        self.guess_beta = []  # Points that can be suggested by the algorithm. Beta values only
        self.final_points = []  # Points that can be propagated by the algorithm
        self.final_beta = []  # Points that can be propagated. Beta values only
        self.observe_points = []  # Random sample of points that could be seen by the algorithm
        self.observe_beta = []  # Random sample of points that could be seen by the algorithm. Beta values only
        self.suggest = []  # Last suggestion done by the algorithm
        self.suggested = []  # Points suggested by the algorithm
        self.current_image = None  # The most recent image (for sampling, etc.)

        self.sample_radius = None  # The radius of the random samples
        self.did_we_sample_the_image = False
        # Initialize ROS message passing
        # self.point_sub = rospy.Subscriber("/gui_points",point,user_points)
        # self.user_obs = rospy.Subscriber("/user_response",Bool,PF_update)
        # self.seg_image = rospy.Subscriber("/seg_image",??,image_sample)
        # self.suggest_pub = rospy.Publisher("/suggest",point,queue_size=1)
        # self.populate_pub = rospy.Publisher("/populate",point)

        # Call and run Julia functions
        Main.include("PE_POMDP_Def.jl")
        Main.include("ParticleFilter_Def.jl")
        self.user = Main.user_expert
        self.solver = Main.POMCPSolver(tree_queries=100, c=100.0, rng=Main.MersenneTwister(1), tree_in_info=True,
                                       estimate_value=Main.FORollout(Main.RandomSolver()))
        print("Algorithm Initialization Complete")

    def new_user_point(self, msg):
        """Function adds new user points to a list.
        Will keep adding items to a list until full number of guesses has been reached"""
        self.user_points_msgs.append(msg)

        # the case that we received the image before the first user point (most likely)
        if self.current_image is not None and self.did_we_sample_the_image is False:
            self.sample_radius = msg.r
            self.sample_image(processing_type.RANDOM)
            self.did_we_sample_the_image = True

        if len(self.user_points_msgs) == self.init_guess:
            # Sample the user specified points
            self.sample_image(processing_type.USER)
            # Reset the message array
            self.user_points_msgs = []
            # Generate a mean vector to initialize particle belief
            self.phi = [sum(x) / len(x) for x in zip(*self.user_points[:, 3:])]
            # Initialize belief
            self.belief = Main.init_PF(self.phi, False, self.num_particles)

    def generate_suggestion(self):
        """Function is given a set of user points and generates a suggestion based on the algorithm
        Engage after user clicks box -- for now..."""
        self.phi = [sum(x) / len(x) for x in zip(*self.belief.states)]
        # Sample set of points to iterate over
        self.best_points_idx, self.best_points_phi = Main.find_similar_points(self.guess_beta, self.phi,
                                                                              self.p_sample, self.suggested)
        # print(list(self.best_points_phi))

        # Generate new suggestion using Julia function call
        suggest = Main.find_next_action(list(self.user_points), self.best_points_phi, self.observe_beta,
                                        self.final_beta, self.user, self.num_guess)

        self.suggested.append(suggest)
        self.num_guess -= 1  # Decrement step counter
        print(suggest)
        # Find index that the algorithm suggested
        # If the algorithm chooses to wait, then don't publish anything
        if suggest != "wait":
            ans = self.guess_points[int(suggest)]
            print("The ans is ", ans)
            # Publish to ROS
        else:
            ans = "wait"
        # self.suggest_pub.publish(ans)
        return ans

    def belief_update(self, msg):
        self.belief = Main.update_PF(self.belief, self.POMDP, self.suggest, msg)
        return self.belief

    def handle_new_image(self, msg):
        """ Function will handle any new image received"""
        self.current_image = msg.img

        # the case that we received the first user point before the image
        if self.sample_radius is not None and self.did_we_sample_the_image is False:
            self.sample_image(processing_type.RANDOM)
            self.did_we_sample_the_image = True

    def sample_image(self, image_processing_type):
        """Function will take in a numpy array for the segmented image and populate necessary lists for sampling"""

        if image_processing_type == processing_type.RANDOM:
            # Update
            # Points that can be suggested by the algorithm
            self.guess_points = image_sampling.sample_class(im=self.current_image, num_samples=self.num_samples,
                                                            r=self.sample_radius)
            # Points that can be propagated by the algorithm
            self.final_points = image_sampling.sample_class(im=self.current_image, num_samples=self.num_samples,
                                                            r=self.sample_radius)
            # Random sample of points that could be seen by the algorithm
            self.observe_points = image_sampling.sample_class(im=self.current_image, num_samples=self.num_samples,
                                                              r=self.sample_radius)
            # Can be used to show sampled points
            # image_sampling.show_save_image(self.current_image, self.guess_points, "", show=True, save=False)
            # Sample beta values for...
            # Points that can be suggested by the algorithm. Beta values only
            self.guess_beta = list(self.guess_points[:, 3:])
            # Points that can be propagated. Beta values only
            self.final_beta = list(self.final_points[:, 3:])
            # Random sample of points that could be seen by the algorithm. Beta values only
            self.observe_beta = list(self.observe_points[:, 3:])

        elif image_processing_type == processing_type.USER:
            tmp_pts = [[m.x, m.y] for m in self.user_points_msgs]
            self.user_points = image_sampling.sample_class(im=self.current_image, sample_points=tmp_pts,
                                                           num_samples=len(tmp_pts), r=self.sample_radius)

    def propagate_belief(self):
        """Applies points onto final image. Actual number of points specified in __init__"""
        points = Main.final_guess(self.final_points, self.belief, self.num_final_guess)
        return points


## MAIN FUNCTION INITIALIZATION SEQUENCE FOR JULIA #####

if __name__ == '__main__':

    p = PreferenceAlgorithm()
    #
    # Testing image sampling
    #
    im_fname = './images/neighborhood_image_segmented.png'
    import cv2.cv2 as cv

    img = cv.imread(im_fname)
    img_msgs = image_sampling.dummy_image_msg(img)

    p.handle_new_image(img_msgs)

    for i in range(5):
        usr_msg = image_sampling.dummy_user_point_msg(
            np.random.randint(25, 475),
            np.random.randint(25, 675),
            50)
        p.new_user_point(usr_msg)
    a = p.generate_suggestion()
    print("Done")
    # print(a)
    # while not rospy.is_shutdown():
    #     rospy.spin()
