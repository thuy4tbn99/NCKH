import pickle
import json

def user_vector():
    file = open('../model/thuytt_ver2.1/uservector_dict.pkl', 'rb')
    userVector_dict = pickle.load(file)
    print('leng userVector dict:',len(userVector_dict))

    response = dict() # response to client

    res_data = [] # save all userid-rep && convert data format
    for idx,userid in enumerate(userVector_dict):
        if(idx < 10):
            res = dict()
            res['userID'] = str(userid)
            res['representation'] = userVector_dict[userid].tolist()
            res_data.append(res)

    response['data'] = res_data
    response2json = json.dumps(response)
    return response2json
    
user_vector()


