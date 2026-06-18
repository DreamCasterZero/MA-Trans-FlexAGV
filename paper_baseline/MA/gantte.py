# import matplotlib
# matplotlib.use('TkAgg')
#
# import matplotlib.pyplot as plt
#
# def Gantt(Machines,agvs=None,file=None):
#     plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
#     plt.rcParams['axes.unicode_minus'] = False  # 显示负号
#     M = ['red', 'blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
#          'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
#          'navajowhite','navy', 'sandybrown', 'moccasin']
#     Job_text=['J'+str(i+1) for i in range(100)]
#     Machine_text=['M'+str(i+1) for i in range(50)]
#     t = 0
#     k=0
#     if agvs!=None:
#         for k in range(len(agvs)):
#             for m in range(len(agvs[k].using_time)):
#                 if agvs[k].using_time[m][1] - agvs[k].using_time[m][0] != 0:
#                     if agvs[k]._on[m]!=None:
#                         plt.barh(k, width= agvs[k].using_time[m][1]- agvs[k].using_time[m][0],
#                                         height=0.6,
#                                         left=agvs[k].using_time[m][0],
#                                         color=M[agvs[k]._on[m]],
#                                         edgecolor='black')
#                     else:
#                         plt.barh(k, width=agvs[k].using_time[m][1] - agvs[k].using_time[m][0],
#                                  height=0.6,
#                                  left=agvs[k].using_time[m][0],
#                                  color='white',
#                                  edgecolor='black')
#                     # plt.text(x=agvs[k].using_time[m][0]+(agvs[k].using_time[m][1] - agvs[k].using_time[m][0])/2-2,
#                     #          y=k-0.05,
#                     #          # s=Machine_text[agvs[k].trace[m]]+'-'+Machine_text[agvs[k].trace[m+1]],
#                     #          fontsize=5)
#                 if  agvs[k].using_time[m][1]>t:
#                     t=agvs[k].using_time[m][1]
#
#     for i in range(len(Machines)):
#         for j in range(len(Machines[i].using_time)):
#             if Machines[i].using_time[j][1] - Machines[i].using_time[j][0] != 0:
#                 plt.barh(i+k+1, width=Machines[i].using_time[j][1] - Machines[i].using_time[j][0],
#                          height=0.8, left=Machines[i].using_time[j][0],
#                          color=M[Machines[i]._on[j]],
#                          edgecolor='black')
#                 plt.text(x=Machines[i].using_time[j][0]+(Machines[i].using_time[j][1] - Machines[i].using_time[j][0])/2 - 0.1,
#                          y=i+k+1,
#                          s=Job_text[Machines[i]._on[j] - 1],
#                          fontsize=12)
#             if Machines[i].using_time[j][1]>t:
#                 t=Machines[i].using_time[j][1]
#     if agvs!=None:
#         list=['AGV1','AGV2']
#         list1=['M'+str(_+1) for _ in range(len(Machines))]
#         list.extend(list1)
#         plt.xlim(0,t)
#         plt.hlines(k + 0.4,xmin=0,xmax=t, color="black")  # 横线
#         # plt.yticks(np.arange(i + k + 3), list,size=13,)
#         plt.title('Scheduling Gantt chart')
#         plt.ylabel('Machines')
#         plt.xlabel('Time(s)')
#     if file!=None:
#         with open(file,'wb') as fb:
#             plt.savefig(fb,dpi=600, bbox_inches='tight')
#     plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('TkAgg')  # 设置后端

def draw_gantt(machines, agvs = None):
    plt.figure(figsize=(10, 3.5), dpi=100)
    M = ['skyblue', 'palegoldenrod', 'lightgreen', 'moccasin', 'lightpink', 'lightgray', 'purple', 'pink', 'Thistle', 'Magenta',
         'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
         'navajowhite', 'navy', 'sandybrown', 'moccasin']
    Job_text = ['J' + str(i + 1) for i in range(100)]
    Machine_text = ['M' + str(i) for i in range(50)]
    t = 0
    k = 0
    if agvs != None:
        for k in range(len(agvs)):
            for m in range(len(agvs[k].using_time)):
                if agvs[k].using_time[m][1] - agvs[k].using_time[m][0] != 0:
                    if agvs[k].on[m]!=None:
                        plt.barh(k, width= agvs[k].using_time[m][1]- agvs[k].using_time[m][0],
                                        height=0.6,
                                        left=agvs[k].using_time[m][0],
                                        color=M[agvs[k].on[m]-1],
                                        edgecolor='black')
                        # plt.text(x=agvs[k].using_time[m][0] + (
                        #             agvs[k].using_time[m][1] - agvs[k].using_time[m][0]) / 2 - 1.5,
                        #          y= k - 0.15,
                        #          s=Job_text[agvs[k].on[m] - 1],
                        #          fontsize=12)
                    else:
                        plt.barh(k, width=agvs[k].using_time[m][1] - agvs[k].using_time[m][0],
                                 height=0.6,
                                 left=agvs[k].using_time[m][0],
                                 color='white',
                                 edgecolor='black')
                if agvs[k].using_time[m][1] > t:
                    t = agvs[k].using_time[m][1]
    for i in range(1,len(machines)):
        for j in range(len(machines[i].using_time)):
            if machines[i].using_time[j][1] - machines[i].using_time[j][0] != 0:
                plt.barh(i+k, width=machines[i].using_time[j][1] - machines[i].using_time[j][0],
                         height=0.8, left=machines[i].using_time[j][0],
                         color=M[machines[i].on[j]-1],
                         edgecolor='black')
                # plt.text(x=machines[i].using_time[j][0]+(machines[i].using_time[j][1] - machines[i].using_time[j][0])/2 - 1.5,
                #          y=i+k-0.15,
                #          s=Job_text[machines[i].on[j]-1],
                #          fontsize=12)
            if machines[i].using_time[j][1] > t:
                t = machines[i].using_time[j][1]

    if agvs!=None:
        list=['AGV' + str(i+1) for i in range(len(agvs))]
        list1=['M'+str(_) for _ in range(1,len(machines))]
        list.extend(list1)
        plt.xlim(0,t)
        plt.ylim(-0.5, len(list) - 0.5)
        plt.yticks(range(len(list)), list)
        plt.hlines(k + 0.4,xmin=0,xmax=t, color="black")  # 横线
        plt.title('Scheduling Gantt chart')
        plt.ylabel('Machines & AGVs')
        plt.xlabel('Time(s)')

    plt.show()
