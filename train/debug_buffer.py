# -*- coding: utf-8 -*-
import numpy as np

from trainers.buffer import Buffer


sequence_length = 64
#sequence_length = 1

time_horizon = 128
#time_horizon = 127


training_buffer = Buffer()
agent_id = 0
actions = [1,2]

for i in range(300):
    training_buffer[agent_id]['actions'].append(actions)
    agent_actions = training_buffer[agent_id]['actions']

    #if len(agent_actions) > time_horizon or i == 150:
    if len(agent_actions) > time_horizon:
        print("agent buffer size={}".format(len(agent_actions)))

        # Updateバッファへコピー
        training_buffer.append_update_buffer(
            agent_id=agent_id,
            batch_size=None,
            training_length=sequence_length)
        # sequence_length=64の場合は、(3,64,2) のバッファになり、
        # sequence_length=1の場合は、(129,2) のバッファになっている.
        # sequence_length=64の場合は、64個に足りない部分はゼロで穴埋めされる.
        update_actions = training_buffer.update_buffer['actions']
        print("update buffer size={}".format(len(update_actions)))

        for j in range(len(update_actions)):
            print(update_actions[j])

        # Agentバッファを全クリアする
        training_buffer[agent_id].reset_agent()

        agent_actions = training_buffer[agent_id]['actions']
        print("agent buffer size={}".format(len(agent_actions)))
        
        #break
