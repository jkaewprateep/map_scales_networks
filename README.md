# map_scales_networks
For study map scales networks, simple implementation with the input of map data into networks using scales manipulation as grids image or logarithms images simply return something as the map variable of the game MonsterKong.

## Variables ##

We created a list of actions to retrieve the action name, look up possible actions, and transform other method inputs such as random or play records from different batch scripts easier. Further, they can return priority values and item types that we use for text string return or image return for a user and logical mapping item distance and priorities.

```
actions = { "none_1": K_h, "left": K_a, "down": K_s, "right": K_d, 
        "up": K_w, "action": K_SPACE }
dic_itemtype = { "player" : 90, "monster" : 91, "coin" : 92, "background" : 93, "fireball" : 94, 
        "ladder" : 95, "wall" : 96, "allies" : 97 }
dic_itempriority = { "player" : 0, "monster" : 4, "coin" : 2, "background" : 0, "fireball" : 4, 
        "ladder" : 9, "wall" : -9, "allies" : 50 }
```

## Load game map into arrayList or varible ##

Simple as in the previous project a map variable can access from many ways including class like variable access.

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

## Files and Directory ##

| File Name | Description  |
--- | --- |
|sample.py|sample codes|
|04.png|Distance attention networks|
|05.png|Map Scales|
|path_order.gif|PATH Finding, simple|
|wizard-of-war.gif|Map Scales, WOW|
|README.md|readme file|

## Result ##


#### Distance attnetion networks ####

![Distance attnetion networks](https://github.com/jkaewprateep/map_scales_networks/blob/main/04.png?raw=true "Distance attnetion networks")

#### Path room ordering ####

![Path ordering](https://github.com/jkaewprateep/map_scales_networks/blob/main/path_order.gif?raw=true "Path ordering")

#### Map scales ####

![Map scales](https://github.com/jkaewprateep/map_scales_networks/blob/main/05.png?raw=true "Map scales")

#### Map scales Wizards of Wars ####

![Map scales WOW](https://github.com/jkaewprateep/map_scales_networks/blob/main/wizard-of-war.gif?raw=true "Map scales")
