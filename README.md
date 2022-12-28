# map_scales_networks
For study map scales networks

```
def	read_current_state( string_gamestate ):
    if string_gamestate in ['score']:
        return game_console.getScore()
    ...
    
    elif string_gamestate in ['map']:
        Map_Game = game_console.newGame.map	# 30x80
        return Map_Game	

```


## Result ##


#### Distance attnetion networks ####

![Distance attnetion networks](https://github.com/jkaewprateep/map_scales_networks/blob/main/04.png?raw=true "Distance attnetion networks")

#### Path room ordering ####

![Path ordering](https://github.com/jkaewprateep/map_scales_networks/blob/main/path_order.gif?raw=true "Path ordering")

#### Map scales ####

![Map scales](https://github.com/jkaewprateep/map_scales_networks/blob/main/05.png?raw=true "Map scales")
