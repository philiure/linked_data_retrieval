# START SERVER
# docker run -p 1491:1491 -v /Users/philippe/Desktop/Sonic_implementation/config.cfg:/etc/sonic.cfg -v /Users/philippe/Desktop/Sonic_implementation/store/:/var/lib/sonic/store/ valeriansaliou/sonic:v1.3.2

from sonic import SearchClient
import json

user_query = 'beatles'
collection = 'TEST'
bucket = 'Default'

json_file = json.load(open('data.json'))

with SearchClient("127.0.0.1", 1491, "SecretPassword") as querycl:
    print(querycl.ping())
    result_query = querycl.query(collection, bucket, user_query)
    print(result_query)
    for id in result_query:
        print(json_file[id]['subject'], json_file[id]['predicate'], json_file[id]['object'])