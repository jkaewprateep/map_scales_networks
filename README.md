# map_scales_networks
For study map scales networks

## Load game map into arrayList or varible ##

```
def read_current_state( string_gamestate ):
    if string_gamestate in ['score']:
        return game_console.getScore()
    ...
    
    elif string_gamestate in ['map']:
        Map_Game = game_console.newGame.map	# 30x80
        return Map_Game	

```

## Sorted order priority and distance for simple PATH finding, can resolved by AI or perfrom sorting by AI ##

```
def update_DATA( action ):
    steps = steps + 1
    map = read_current_state("map")
    
    list_ladder = elementListCreate( ladders, 95, bRelativePlayer=True )
    list_ladder.sort(key=elementDistancePhase)
```

## Controls functions, objectives ##

```
contrl = ( 1800 - steps ) - gamescores + ( 100 * reward )
contr2 = lives
contr3 = steps - player[0][0] * ( 3 * ( steps % 3 ) - 4 )
```

## Result ##


#### Distance attnetion networks ####

![Distance attnetion networks](https://github.com/jkaewprateep/map_scales_networks/blob/main/04.png?raw=true "Distance attnetion networks")

#### Path room ordering ####

![Path ordering](https://github.com/jkaewprateep/map_scales_networks/blob/main/path_order.gif?raw=true "Path ordering")

#### Map scales ####

![Map scales](https://github.com/jkaewprateep/map_scales_networks/blob/main/05.png?raw=true "Map scales")
