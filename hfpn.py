import numpy as np

def attention_layer(feature_map):
    channels, height, width = feature_map.shape
    attention_map = np.zeros((channels, height, width))
    for c in range(channels):
        max_value = np.max(feature_map[c])
        attention_map[c] = feature_map[c] / max_value  # normalize
    return attention_map

def amplify_low_level_features(low_level_feature_map):
    amplified_feature_map = low_level_feature_map ** 2
    return amplified_feature_map

def occlusion_sensitive_fusion(features1, features2):
    # resize both spatial and channel dimensions to match
    channels, height, width = features1.shape
    resized_features2 = np.resize(features2, (channels, height, width))
    
    # perform element-wise maximum operation
    fused_features = np.maximum(features1, resized_features2)
    return fused_features


def hybrid_fpn(feature_pyramids):
    num_levels = len(feature_pyramids)
    enhanced_pyramids = []
    
    for i in range(num_levels):
        feature_map = feature_pyramids[i]
        attention_map = attention_layer(feature_map)
        
        if i == 0:
            attention_map = amplify_low_level_features(attention_map)
        
        if i > 0:
            enhanced_feature = occlusion_sensitive_fusion(enhanced_pyramids[i-1], attention_map)
        else:
            enhanced_feature = attention_map
        
        enhanced_pyramids.append(enhanced_feature)
    
    return enhanced_pyramids
