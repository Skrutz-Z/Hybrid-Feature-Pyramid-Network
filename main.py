
from hfpn import hybrid_fpn
from utils import load_sample_data, visualize_prediction

#random feature pyramid as example
pyramid_input = load_sample_data()

#Hybrid Feature Pyramid Network
hf_pyramids = hybrid_fpn(pyramid_input)

#output
visualize_prediction(hf_pyramids)
