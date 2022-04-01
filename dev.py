# version 5 rotated animation

def Plot3DVersion5(elevation = 20., azimuth = -35, flag = 1, i = 0):

  if( flag != 0 ):
    szin = np.arange(len(auto.sensor_right))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(auto.sensor_left, auto.sensor_right, auto.y_distance, c=szin)
    ax.set_xlabel('sensor left')
    ax.set_ylabel('sensor right')
    ax.set_zlabel('y_distance')
    # ax.invert_xaxis()

    __x_min = 0; __x_max = 200;
    __y_min = 0; __y_max = 200;
    __z_min = -50; __z_max = 50;

    ax.set_xlim(__x_min, __x_max);
    ax.set_ylim(__y_min, __y_max);
    ax.set_zlim(__z_min, __z_max);

    __x = auto.sensor_left[-1]; __y = auto.sensor_right[-1]; __z = auto.y_distance[-1];

    xe = __x; xv = __x;
    ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
    ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

    xe = __x_min; xv = __x_max;
    ye = __y; yv = __y;
    ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

    xe = ax.get_xlim()[0]; xv = ax.get_xlim()[0];
    ye = __y; yv = __y;
    ze = __z_min; zv = __z_max;

    ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

    xe = __x_min; xv = __x_min;
    ye = __y_min; yv = __y_max;
    ze = __z; zv = __z;

    ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

    xe = __x_min; xv = __x_max;
    ye = __y_max; yv = __y_max;
    ze = __z; zv = __z;

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

    xe = __x; xv = __x;
    ye = __y_max; yv = __y_max;
    ze = __z_min; zv = __z_max;

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

    # ---
    xe = __x; xv = __x;
    ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
    ze = __z; zv = __z;

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange', linestyle='dashed') #dotted

    xe = __x_min; xv = __x;
    ye = __y; yv = __y;
    ze = __z; zv = __z;

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='green', linestyle='dashed') #dotted

    xe = __x; xv = __x;
    ye = __y; yv = __y;
    ze = __z_min; zv = __z;

    ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue', linestyle='dashed') #dotted

    ax.view_init(elev=elevation, azim=azimuth)

    if( flag == 1 or flag == 3 ): plt.show();
    if( flag == 2 or flag == 3 ): fig.savefig('Plot3D_{0:04}'.format(i)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');