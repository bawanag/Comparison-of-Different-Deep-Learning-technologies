import numpy as np
import scipy.stats as st
import math
class calculator:
        def __init__(self):
                super().__init__()

        
        def mean(self,iterable=[]):
                a = np.array(iterable)
                return np.mean(a).astype(np.float32)

        def mean_interval(self,confident_level_rate = 0.90, arraylist = None):
                currentmean = self.mean(arraylist)
                interval_result=st.norm.interval(confident_level_rate,currentmean,st.sem(arraylist))
                if math.isnan(interval_result[0]):
                        interval_result = [currentmean,currentmean]
                return interval_result

        def geo_mean(self,iterable=[]):
                a = np.array(iterable)
                return a.prod()**(1.0/len(a))

        def geo_mean_interval(self,confident_level_rate = 0.90, arraylist = None):
                geomean = self.geo_mean(arraylist)
                interval_result = st.t.interval(confident_level_rate, len(arraylist)-1, loc=geomean, scale=st.sem(arraylist))
                if math.isnan(interval_result[0]):
                        interval_result = [geomean,geomean]
                return interval_result
                
if __name__ == "__main__":
        # a = [1,1,1,1,1,1]
        a = [1,2,3,4,5,6,7,8,9,10]
        # a= [2196.09375, 2196.27734375, 2204.8203125, 2200.86328125, 2215.88671875, 2198.21484375, 2202.23046875, 2202.42578125, 2196.25, 2213.45703125, 2196.3046875, 2202.6171875, 2199.53125, 2196.23046875, 2198.3046875, 2207.25390625, 2197.4921875, 2200.11328125, 2196.28515625, 2209.28125]
        c = calculator()
        print(c.geo_mean(a))
        print(c.geo_mean_interval(confident_level_rate = 0.9, arraylist = a))
        print(c.mean(a))
        print(c.mean_interval(confident_level_rate = 0.9, arraylist = a))
