Please start Sonic server

Docker method:

$ docker run -p 1491:1491 -v /Path/to/config/Sonic Example/config.cfg:/etc/sonic.cfg -v /Path/to/store/Sonic Example/store/:/var/lib/sonic/store/ valeriansaliou/sonic:v1.3.2