# START SEVER:
# docker run -p 1491:1491 -v /Users/philippe/Desktop/Sonic_implementation/config.cfg:/etc/sonic.cfg -v /Users/philippe/Desktop/Sonic_implementation/store/:/var/lib/sonic/store/ valeriansaliou/sonic:v1.3.2

from sonic import IngestClient
import json
collection = 'TEST'
bucket = 'Default'

json_file = json.load(open('data.json'))

# print(json_file.keys()) #['1']['description']

with IngestClient("127.0.0.1", 1491, "SecretPassword") as ingestcl:
    print(ingestcl.ping())
    print(ingestcl.protocol)
    print(ingestcl.bufsize)
    for id in json_file.keys():
        ingestcl.push(collection, bucket, id, json_file[id]['description'])