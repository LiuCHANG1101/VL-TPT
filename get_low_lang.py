import rlbench.utils as rlbench_utils
import hydra
from omegaconf import DictConfig
import os
from rlbench.backend.const import *
import pickle
from helpers import demo_loading_utils
from pathlib import Path
import json

def get_low_lang(datapath:str) -> None:
    datapath = Path(datapath)
    finish_task=[]
    for path in datapath.rglob('low_dim_obs*'):
        get_each_lang(path,finish_task)
    pass

def get_each_lang(datapath,finish_task):
    low_lang_path =  os.path.join(datapath.parent,"low_lang.pkl")
    task_name = str(datapath.parents[3].name)
    variation_name = str(datapath.parents[2].name)[-1]
    # variation = datapath.parents[2]
    # if variation in finish_task or task_name in low_lang_template.keys():
    #     return 
    with open(datapath, 'rb') as f:
        obs = pickle.load(f)
    if "peract-main" in str(datapath):
        episode_descriptions = os.path.join(datapath.parent, VARIATION_DESCRIPTIONS)
        if os.path.exists(episode_descriptions):
            with open(episode_descriptions, 'rb') as f:
                descriptions_list = pickle.load(f)
                descriptions = descriptions_list[0]   
    elif "template" in str(datapath):
        with open(os.path.join(datapath.parents[4],'instructions.json')) as f:
            try:
                descriptions = json.load(f)[task_name][variation_name][0]
            except:
                return
    low_lang_template = {

    "open_drawer":["Move to the front of the handle of the {target_object} ","Grasp the handle of the {target_object}","Pull the {target_object}"],
    "close_drawer":["Move to the front of the opened {target_object} ","Push the {target_object} to close it"],
    "open_door":["Move to the front of the handle of the door; Grasp the handle of the door.", "Rotate around the axis of the handle joint.", "Rotate around the axis of the door joint ."],
    'close_jar':["Move to the top of the jar lid","Grasp the jar lid","Lift the jar lid","Move to the top of {target_object}","Put the jar lid on the {target_object}","Rotate the jar lid"],
    'put_item_in_drawer':["Move to the top of the drawer","Move to the front of the {target_place}","Grasp the handle of the {target_place}","Pull the {target_place}","Move away from the grasping {target_place}","Move to the top of the opened drawer","Move to the top of the block","Grasp the block","Lift the block","Move to the top of the opened drawer","put the item in the {target_place}"],
    'put_groceries_in_cupboard':["Move to the top of {target_object}","Grasp the {target_object}","Lift the {target_object}","Align the {target_object} with cupboard","put {target_object} in cupboard"],
    "put_money_in_safe":["Move to the front of the money","Grasp the money","Move the money away","Align the money with the {target_place}","put the money on the {target_place}"],
    'meat_off_grill':["Move to the top of the {target_object}","Grasp the {target_object}","Lift the {target_object}","Move to the top of desk",'put the {target_object} on the desk'],
    'place_shape_in_shape_sorter':["Move to the top of the {target_object}","Grasp the {target_object}","Lift the {target_object}","Align the {target_object} with the {target_object} shape sorter","put the {target_object} in the {target_object} sorter"],
    'place_wine_at_rack_location':["Move to the front of the wine","Grasp the wine","Lift the wine","Rotate the wine to horizontal","Put the wine bottle to {target_place} "],
    'stack_blocks':["Move to the top of the {target_object}","Grasp the {target_object}","Lift the {target_object}","Align the block with the {target_object}","Place the block onto the {target_object}","Move to the top of the {target_object}"],
    'slide_block_to_color_target':["Move to the edge of the block","slide the block to {target_place}"],
    "place_cups":["Move to the top of cup","Grasp the cup","Lift the cup","Place the cup on the cup holder","Place the cup on the cup holder","Place the cup on the cup holder","Place the cup on the cup holder"],
    'stack_cups':["Move to the top of cup beside {target_place}","Grasp the cup","Lift the cup","Align the cup with the {target_place}","stack the cup on top of the {target_place}","Move to the top of cup beside {target_place}","Grasp the cup","Lift the cup","Align the cup with the {target_place}","stack the cup on top of the {target_place}"],
    'scoop_with_spatula':["Move to the handle of a spatula","Grasp the handle of a spatula","Scoop up the cube with spatula","Lift up the cube with spatula"],
    "take_usb_out_of_computer":["Grasp the USB","Take out the USB"],
    'take_frame_off_hanger':["Grasp the frame","Move along the the axsis of the hanger","Take out the frame from the hanger","Put the frame on the table"],
    'hang frame on hanger':["Grasp the frame","Lift the frame","Align the frame with the hanger","Hang frame on hanger"],
    'put_shoes_in_box':[],
    'press_switch':["Move to the switch","Press the switch"],
    'turn_tap':["Move to the {target_object}","Turn on the {target_object}"],
    'place_hanger_on_rack':["Move to the hanger","Grasp the hanger","Lift the hanger","Move the hanger to the rack","Place the hanger on the rack"],
    'take_tray_out_of_oven':[],
    'take_shoes_out_of_box':[],
    'hit_ball_with_queue':["Move to the cue","Grasp the cue","Lift the cue","Align the cue with the ball",'Hit ball with cue in to the goal','Hit ball with cue in to the goal','Hit ball with cue in to the goal'],
    'meat_on_grill':["Move to the top of the {target_object}","Grasp the {target_object}","Lift the {target_object}","Move to the top of desk",'Put the {target_object} on the desk'],
    'push_repeated_buttons':[],
    'open_fridge':["Move to the front of the handle of the fridge","Grasp the handle of the  the fridge","Open the fridge"],
    'close box':["Move to the front of the opened box","Grasp the box edge","Rotate the box edge","Release the gripper","Move away from the box"],
    'solve_puzzle':["Move to the top of the piece","Grasp the piece","Lift the piece","Move to the top of the puzzle","Align the piece with the puzzle","Release the gripper"],
    'take_umbrella_out_of_umbrella_stand':["Move to the top of the umbrella","Grasp the rear of the umbrella","Lift the umbrella"],
    'take toilet roll off stand':["Move to the front of the toilet roll","Grasp the toilet roll","Take out the the toilet roll"],
    'reach_and_drag':["Move to the top of stick","Grasp the stick","Lift the stick","Align the stick with the cube","Put the stick on the edge of the cude","Move the cube to the {target_place}"],
    "take_plate_off_colored_dish_rack":["Move to the top of {target_object}","Grasp the plate","Lift the plate","Move to the top of box","Place the plate onto the box"],
    'insert_onto_square_peg':["Move to the top of ring","Grasp the ring","Lift the ring","Move to the top of {target_place}","Put the ring on the {target_place}"],
    'light_bulb_in':["Move to the top of {target_object}","Grasp the {target_object}","Lift the {target_object}","Move to the top of the lamp","Put the {target_object} to the lamp","Rotate the {target_object}","Move to the top of the lamp and Release"],
    "push_buttons":["Move to the top of {target_1}","Push {target_1}","Move to the top of {target_2}","Push {target_2}","Move to the top of {target_3}","Push {target_3}"],
    "sweep_to_dustpan_of_size":["Move to front of the handle of the broom","Grasp the broom","Move to the top of the dust","Align the broom with the dust","sweep the dust to {target_place}"],

    }
    locations={
        'reach_and_drag':{
            "target_place":[-2,-1]
        },
        "block_pyramid":{
            "target_object":[1,2],
        },
        "change_channel":{
            "target_object":[3],
        },
        "close_drawer":{
            "target_object":[1,2],
        },
        "close_jar":{
            "target_object":[2,3],
        },
        "close_fridge":{
            "target_object":[1],
        },
        "empty_container":{
            "target_place":[-2,-1]
        },
        "hang_frame_on_hanger":{
            "target_object":[1],
            "target_place":[3]
        },
        "insert_onto_square_peg":{
            "target_place":[-2,-1]
        },
        "lift_numbered_block":{
            "target_object":[7],
        },
        "light_bulb_out":{
            "target_object":[-2,-1],
        },
        "light_bulb_in":{
            "target_object":[-3,-2,-1],
        },
        "meat_on_grill":{
            "target_object":[2],
        },
        "meat_off_grill":{
            "target_object":[2],
        },
        "open_drawer":{
            "target_object":[2,3],
        },
        "open_jar":{
            "target_object":[-2,-1],
        },
        "push_button":{
            "target_object":[-2,-1],
        },
        "push_buttons":{
            "target_1":[2,3],
            "target_2":[7,8],
            "target_3":[-2,-1]
        },
        "pick_and_lift":{
            "target_object":[3,4],
        },
        "pick_and_lift_small":{
            "target_object":[3],
        },
        "pick_up_cup":{
            "target_object":[-2,-1],
        },
        "pour_from_cup_to_cup":{
            "target_object":[4,5],
            "target_place":[-2,-1]
        },
        "push_repeated_buttons":{
            "target_1":[2,3],
            "target_2":[7,8],
            "target_3":[-2,-1]
        },
        "put_books_on_bookshelf":{
            "target_object":[1,2],
        },
        "put_groceries_in_cupboard":{
            "target_object":[2],
        },
        "put_tray_in_oven":{
            "target_object":[1],
            "target_place":[3]
        },
        "put_item_in_drawer":{
            "target_place":[-2,-1]
        },
        "put_all_groceries_in_cupboard":{
            "target_object":[4],
            "target_place":[7]
        },
        "put_knife_in_knife_block":{
            "target_object":[2],
            "target_place":[5,6]
        },
        "put_money_in_safe":{
            "target_object":[-2,-1],
        },
        "put_rubbish_in_bin":{
            "target_object":[1],
            "target_place":[3]
        },
        "put_toilet_roll_on_stand":{
            "target_object":[1,2],
            "target_place":[4]
        },
        "place_cups":{
            "target_number":[1],
            "target_place":[-2,-1]
        },
        "place_shape_in_shape_sorter":{
            "target_object":[2],
        },
        "reach_target":{
            "target_object":[-2,-1],
        },
        "reach_and_drag":{
            "target_place":[-2,-1]
        },
        "remove_cups":{
            "target_object":[1,2],
        },
        "setup_checkers":{
            "target_object":[2,3],
        },
        "setup_chess":{
            "target_object":[2,3],
        },
        "stack_chairs":{
            "target_place":[8,9]
        },
        "stack_cups":{
            "target_place":[-2,-1]
        },
        "stack_wine":{
            "target_object":[1,2],
        },
        "slide_cabinet_open":{
            "target_object":[1,2],
        },
        "slide_cabinet_open_and_place_cups":{
            "target_place":[-2,-1]
        },
        "take_plate_off_colored_dish_rack":{
            "target_object":[-3,-2,-1],
        },
        "take_money_out_safe":{
            "target_object":[6,7],
        },
        "turn_tap":{
            "target_object":[1,2],
        },
        "tv_on":{
            "target_object":[3],
        },
        "take_item_out_of_drawer":{
            "target_object":[-2,-1],
        },
        "take_cup_out_from_cabinet":{
            "target_place":[6,7,8,9,10]
        },
        "take_shoes_out_of_box":{
            "target_object":[1],
            "target_place":[4]
        },
        "take_frame_off_hanger":{
            "target_object":[1],
            "target_place":[3]
        },
        "turn_oven_on":{
            "target_object":[3],
        },
        "turn_taps":{
            "target_object":[1,2],
        },
        "unplug_charger":{
            "target_object":[1],
        },
        "weighing_scales":{
            "target_object":[-2,-1],
        },

        "close_drawer":{
            "target_object":[1,2],
        },
        "reachand_drag":{

        },
        'scoop_with_spatula':{

        },
        'insert_onto_square_peg':{
            "target_place":[-2,-1]
        },
        "sweep_to_dustpan_of_size":{
            "target_place":[-2,-1]
        },
        'place_wine_at_rack_location':{
            "target_place":[-5,-4,-5,-2,-1]
        },
        'slide_block_to_color_target':
        {
            "target_place":[-2,-1]
        },
        'put_money_in_safe':{
            "target_place":[-2,-1]
        },
        'put_item_in_drawer':{
            "target_place":[-2,-1]
        },
        'stack_blocks':{
            "target_number":[1],
            "target_object":[2,3]
        }
    }

    episode_keypoints = demo_loading_utils.keypoint_discovery(obs, method='heuristic')
    if episode_keypoints[0]==1:            
        episode_keypoints = episode_keypoints[1:]
    valid = (max_episode_length[0][task_name] == len(episode_keypoints))
    # return
    # keypoint_number = len(episode_keypoints)
    # print(keypoint_number)
            # print(keypoint_number)

    
    location = locations[task_name]
    target_object = []
    target_place = []
    target_multi_object=[]
    for target,loc in location.items():
        multi_object=[]
        for index in loc:
            if index < len(descriptions.split(" ")):
                if target == "target_object":
                    target_object.append(descriptions.split(" ")[index])
                elif target == "target_place":
                    target_place.append(descriptions.split(" ")[index])
                else :
                    multi_object.append(descriptions.split(" ")[index])
        target_multi_object.append(" ".join(multi_object))
    target_object = " ".join(target_object)
    target_place = " ".join(target_place)


    low_lang = low_lang_template[task_name]
    if len(low_lang) == 0:
        low_lang = [descriptions] * len(episode_keypoints)
        pass                   
    else:
        i = 0
        while (len(low_lang))<len(episode_keypoints):
            if "target_number" in location.keys():
                low_lang.append(low_lang_template[task_name][i%len(low_lang_template[task_name])])
                i+=1
            else:
                for i in range(len(episode_keypoints)-1):
                    if episode_keypoints[i+1]-episode_keypoints[i] < 5 or episode_keypoints[i] < 5:
                        low_lang.insert(i,low_lang[i])
                        if (len(low_lang))==len(episode_keypoints):
                            break
                while (len(low_lang))<len(episode_keypoints):
                    low_lang.append(low_lang[-1])
        if (len(low_lang))>len(episode_keypoints):
            low_lang = low_lang[:len(episode_keypoints)]
    if "target_object" in location.keys() or "target_place" in location.keys():
        for i in range(len(low_lang)):
            low_lang[i]=low_lang[i].replace("{target_object}", target_object)
            low_lang[i]=low_lang[i].replace("{target_place}", target_place)
            pass
    elif "target_1" in location.keys():
        for i in range(len(low_lang)):
            low_lang[i]=low_lang[i].replace("{target_1}", target_multi_object[0])
            low_lang[i]=low_lang[i].replace("{target_2}", target_multi_object[1])
            low_lang[i]=low_lang[i].replace("{target_3}", target_multi_object[2])
        low_lang = low_lang[:len(episode_keypoints)]
    if (len(low_lang))!=len(episode_keypoints):
        raise ValueError("长度不匹配")
    print(low_lang)
    with open(low_lang_path,"wb+") as f:
        pickle.dump(low_lang,f)

def get_low_lang_list(datapath:str):
    return  list(Path(datapath).rglob('low_lang*'))

def get_low_lang_gpt(datapath:str) -> None:
    Actions = [
        "Move(object,relative_direction)",
        "Grasp(object)",
        "Lift(object)",
        "Put(object,object)",
        "Rotate(object)",
        "Pull(object)",
        "Push(object)",
        "Align(object,object)",
    ]
    relative_direction  = [
        "top",
        "front",
        "none",
        "away from"
    ]

    max_episode_length={
        "close_jar": 6,
        "insert_onto_square_peg": 5,
        "light_bulb_in": 7,
        "meat_off_grill": 5,
        "open_drawer": 3,
        "reach_and_drag": 6,
        "place_shape_in_shape_sorter": 5,
        'place_wine_at_rack_location': 5,
        "put_item_in_drawer": 11,
        "put_money_in_safe": 5,
        "put_groceries_in_cupboard": 5,
        "stack_cups":10,
        "push_buttons": [2,4,6],
        "sweep_to_dustpan_of_size": 5,
        "turn_tap": 2
    },
    
# "push_buttons": [2,4,6],
# "slide_block_to_color_target": [2,5],
# "place_cups": [6,13,21],
# "stack_blocks": [17,11,23],

    prompts= '''
    # open the top drawer
    think: The target object is the top drawer, and the target place is none. The task "open_drawer" has a maximum episode length of 3.

    do:
    robot.Move(top drawer, front)
    robot.Grasp(top drawer)
    robot.Pull(top drawer)
    done()

    # pick up the lid from the table and put it on the white jar'
    think: The target object is the jar lid, and the target place is the white jar. The task "close_jar" has a maximum episode length of 7.

    do:
    robot.Move(jar lid, top)
    robot.Grasp(jar lid)
    robot.Lift(jar lid)
    robot.Move(white jar, top)
    robot.Put(jar lid, white jar)
    robot.rotate(jar lid)
    done()

    # put the ring on the white spoke
    think: The target object is the ring, and the target place is the white spoke. The task "insert_onto_square_peg" has a maximum episode length of 5.

    do:
    robot.Move(ring, top)
    robot.Grasp(ring)
    robot.Lift(ring)
    robot.Move(white spoke, top)
    robot.Put(ring, white spoke)
    done()

    # put the light bulb from the maroon casing into the lamp
    think: The target object is light bulb from the maroon casing, and the target place is the lamp. The task "light_bulb_in" has a maximum episode length of 7.

    do:
    robot.Move(maroon light bulb, top)
    robot.Grasp(maroon light bulb)
    robot.Lift(maroon light bulb)
    robot.Move(lamp, top)
    robot.put(maroon light bulb, lamp)
    robot.rotate(maroon light bulb)
    robot.Move(lamp, top)
    done()

    # take the chicken off the grill
    think: The target object is the chicken, and the target place is none. The task "meat_off_grill" has a maximum episode length of 5.

    do:
    robot.Move(chicken, top)
    robot.Grasp(chicken)
    robot.Lift(chicken)
    robot.Move(desk, top)
    robot.Put(chicken, desk)
    done()

    # place 3 cups on the cup holder
    think: The target object is the cups, and the target place is the cup holder. The task "place_cups" has a maximum episode length of 6.

    do:
    robot.Move(cups, none)
    robot.Grasp(cups)
    robot.Move(cups, front)
    robot.Put(cups, cup holder)
    done()

    # put the cylinder in the shape sorter
    think: The target object is the cylinder, and the target place is the shape sorter. The task "place_shape_in_shape_sorter" has a maximum episode length of 5.

    do:
    robot.Move(cylinder, top)
    robot.Grasp(cylinder)
    robot.Lift(cylinder)
    robot.Move(shape sorter, top)
    robot.Put(cylinder, shape sorter)
    done()

    # push the maroon button
    think: The target object is the maroon button, and the target place is none. The task "push_buttons" has a maximum episode length of [2,4,6],base on the instruction,the length should be 2.

    do:
    robot.Move(maroon button, top)
    robot.Push(maroon button)
    done()

    # put the soup in the cupboard
    think: The target object is the soup, and the target place is the cupboard. The task "put_groceries_in_cupboard" has a maximum episode length of 5.

    do:
    robot.Move(soup, top)
    robot.Grasp(soup)
    robot.Lift(soup)
    robot.Move(cupboard, top)
    robot.Put(soup, cupboard)
    done()

    # open the the top drawer and put the block in 
    think: The target object is the block, and the target place is the top drawer. The task "put_item_in_drawer" has a maximum episode length of 11.

    do:
    robot.Move(top drawer, top)
    robot.Move(top drawer, front)
    robot.Grasp(top drawer)
    robot.pull(top drawer)
    robot.Move(top drawer, away from)
    robot.Move(top drawer, top)
    robot.Move(block, top)
    robot.Grasp(block)
    robot.Lift(block)
    robot.Move(top drawer, top)
    robot.Put(block, top drawer)
    done()

    # put the money away in the safe on the top shelf
    think: The target object is the money, and the target place is the top shelf. The task "put_money_in_safe" has a maximum episode length of 5.

    do:
    robot.Move(money, front)
    robot.Grasp(money)
    robot.Move(top shelf, front)
    robot.Put(top shelf,none)
    robot.Put(money, top shelf)
    done()

    # use the stick to drag the cube onto the magenta target
    think: The target object is the cube, and the target place is the magenta target. The task "reach_and_drag" has a maximum episode length of 6.

    do:
    robot.Move(stick, top)
    robot.Grasp(stick)
    robot.Lift(stick)
    robot.Align(stick,cube)
    robot.Put(stick,cube)
    robot.Move(cube,magenta target)
    done()

    # grasp the right tap and turn it
    think: The target object is the right tap, and the target place is none. The task "turn_tap" has a maximum episode length of 2.

    do:
    robot.Move(right tap, none)
    robot.rotate(right tap)
    done()

    # stack the wine bottle to the left of the rack
    think: The target object is the wine bottle, and the target place is to the left of the rack. The task "stack_wine" has a maximum episode length of 6.

    do:
    robot.Move(wine bottle, top)
    robot.Grasp(wine bottle)
    robot.Lift(wine bottle)
    robot.Rotate(wine bottle)
    robot.Put(wine bottle, left of rack)
    done()

    # slide the block to blue target
    think: The target object is the block, and the target place is the blue target. The task "slide_block_to_target" has a maximum episode length of 2.

    do:
    robot.Move(block, none)
    robot.Align(block, blue target)
    done()

    # use the broom to brush the dirt into the short dustpan
    think: The target object is broom, and the target place is the short dustpan. The task "sweep_to_dustpan" has a maximum episode length of 5.

    do:
    robot.Move(broom, front)
    robot.Grasp(broom)
    robot.Move(dirt, top)
    robot.Align(broom, dirt)
    robot.Put(dirt, short dustpan)
    done()
    
    '''
    

max_episode_length={
    "close_jar": 6,
    "insert_onto_square_peg": 5,
    "light_bulb_in": 7,
    "meat_off_grill": 5,
    "open_drawer": 3,
    "reach_and_drag": 6,
    "place_shape_in_shape_sorter": 5,
    "place_cups": [6,13,21],
    'place_wine_at_rack_location': 5,
    "put_item_in_drawer": 11,
    "put_money_in_safe": 5,
    "put_groceries_in_cupboard": 5,
    "stack_blocks": [17,11,23],
    "stack_cups":10,
    "push_buttons": [2,4,6],
    "slide_block_to_color_target": [2,5],
    "sweep_to_dustpan_of_size": 5,
    "turn_tap": 2
},

if __name__ == '__main__':
    tasks= ["close_jar"]
    # tasks=["close_jar","insert_onto_square_peg","light_bulb_in","stack_cups","meat_off_grill","open_drawer","place_shape_in_shape_sorter","push_buttons","put_groceries_in_cupboard","put_item_in_drawer","put_money_in_safe","reach_and_drag","turn_tap","place_wine_at_rack_location","sweep_to_dustpan_of_size"]
    # ["close_jar","insert_onto_square_peg","light_bulb_in","stack_cups","meat_off_grill","open_drawer","place_shape_in_shape_sorter","push_buttons","put_groceries_in_cupboard","put_item_in_drawer","put_money_in_safe","reach_and_drag","turn_tap","place_wine_at_rack_location","sweep_to_dustpan_of_size"]
    for task in tasks: 
        get_low_lang("/home/liuchang/projects/peract-main/data/train/"+task)
        # get_low_lang("/home/liuchang/DATA/template_data/0")