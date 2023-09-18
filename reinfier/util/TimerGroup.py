import time
import json
import copy

class TimerGroup:
    def __init__(self,names:list):
        self.names = [name for name in names]
        self.reset()
    
    def reset(self):
        self.names_len=len(self.names)
        self.timevals={name:[] for name in self.names}
        self.name_id = {self.names[i]:i for i in range(self.names_len)}
        self.curr_id = 0
        self.curr_name=self.names[self.curr_id]

        self.prev_val=0
        self.stime = 0

        self.total_time = 0

    
    def start(self,name:str=None):
        if name is not None:
            self.curr_name=name
            self.curr_id = self.name_id[name]
        self.stime = time.time()
        self.total_time = 0


    def switch(self,name:str=None):
        timeval=time.time()-self.stime
        self.timevals[self.curr_name].append(timeval)
        self.total_time +=timeval
        if name is None:
            self.curr_id = (self.curr_id + 1)%self.names_len
            self.curr_name = self.names[self.curr_id]
        else:
            self.curr_name = name
            self.curr_id = self.name_id[name]
        self.stime = time.time()
        return timeval
    
    def stop(self):
        timeval=time.time()-self.stime
        self.timevals[self.curr_name].append(timeval)
        self.total_time +=timeval

        self.stime=time.time()



    def pause(self):
        self.prev_val = time.time() - self.stime

    
    def resume(self):
        self.stime = time.time()-self.prev_val
        self.prev_val = 0

    
    def now(self):
        if self.prev_val == 0:
            timeval=time.time() - self.stime
            return self.curr_name, timeval, self.total_time+timeval
        else:
            return self.curr_name, self.prev_val,self.total_time+self.prev_val

    def get(self,names:list=None,index:int=-1):
        try:
            if names is None:
                names=self.names

            return {name:self.timevals[name][index] for name in names}
        
        except Exception as e:
            print(e)
            return None
    
    def get_all(self):
        return self.timevals

    def dump(self, path:str="timegroup.log",freeze=False):
        if freeze:
            timeval=time.time() - self.stime
            timevals=copy.deepcopy(self.timevals)
            timevals[self.curr_name].append(timeval)
            total_time = self.total_time+timeval

        else:
            timevals=self.timevals

        with open(path, 'w') as f:
            json.dump(timevals, f, indent=4)
        
