import os,sys
import numpy as np
import pandas as pd
import matplotlib
import itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D 
plt.rc('font',  size=20) #, family='serif'


class BinaryElementRadiationDamge:
    def __init__(self):
        pass

    #data: (X,Y,Z)
    def NumDefects(self,data,saveto='results.png',labels=['X','Y']):
        (x,y,Z) = data
        x.append(x[-1]+(x[-1]-x[-2]))
        y.append(y[-1]+(y[-1]-y[-2]))
        x = np.array(x)
        y = np.array(y)
        X, Y = np.meshgrid(x, y)
        z_min, z_max = Z.min(), Z.max()
        plt.figure(figsize=(8,6))
        #plt.subplot(1,1,1)
        #plt.pcolormesh(X, Y, Z, cmap='hsv', vmin=z_min, vmax=z_max)
        plt.pcolor(X, Y, Z, vmin=z_min, vmax=z_max)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        #plt.title('Color by number of potential energy')
        #print X.shape,Y.shape,Z.shape
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.colorbar()
        
        
        
        #plt.xticks(ticks, fontsize = 20)
        #plt.gca().set_xticks(x+w/2)
        #plt.gca().set_xticklabels(tick_labels)
        plt.tight_layout()
        plt.savefig(saveto,dpi=300)
        plt.close()

def TwoVariable(data,saveto='results.png',labels=['X','Y'],legend=None):
    plt.figure(figsize=(8,6))
    (X,Z) = data
    X = np.array(X)
    Z = np.array(Z)
    plt.subplot(1,1,1)
    #plt.pcolormesh(X, Y, Z, cmap='hsv', vmin=z_min, vmax=z_max)
    plt.plot(X, Z,'b-o', linewidth=2,label=legend)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    #plt.title('Color by number of potential energy')
    #plt.axis([x.min(), x.max(), y.min(), y.max()])
    #plt.ylim([-482000,-470000])
    plt.legend(loc='best', shadow=True,prop={'size':14,'weight':'bold'})
    plt.tight_layout()
    #plt.grid()
    plt.savefig(saveto,dpi=300)
    plt.close()

def MultipleTwoVariable(data,saveto='results.png',labels=['X','Y'],legend=None,with_marker=True):
    plt.figure(figsize=(8,6))
    colors = matplotlib.cm.hsv(np.linspace(0.4,1,len(data)))
    markers = itertools.cycle(('o', 'd', 's', '<', '>', 'x')) 
    linestyles = itertools.cycle(('-','-'))
    #colors = itertools.cycle(('r','g','r','g'))
    for i in range(len(data)):
        (X,Z) = data[i]
        X = np.array(X)
        Z = np.array(Z)
        label = None
        marker = None
        linestyle = linestyles.next()
        color=colors[i]
        #color=colors.next()
        if with_marker:
            marker = markers.next()
        if legend is not None:
            label = legend[i]
        plt.plot(X,Z,color=color,marker=marker,markersize=10,linestyle=linestyle,label=label,lw=2)
    #plt.axvline(0.94050393,color='b',linestyle='--',linewidth=2)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    #plt.yscale('log')
    plt.ylim([0.012,0.024])
    #plt.xlim([250,1050])
    plt.legend(loc='lower right', shadow=False,prop={'size':12})
    plt.tight_layout()
    plt.grid()
    plt.savefig(saveto,dpi=300)
    plt.close()


def MultipleTwoVariableErrorBar(data,saveto='results.png',labels=['X','Y'],legend=None,with_marker=True):
    plt.figure(figsize=(8,6))
    colors = matplotlib.cm.hsv(np.linspace(0.4,1,len(data)))
    markers = itertools.cycle(('o', 'd', 's', '<', '>', 'x')) 
    for i in range(len(data)):
        (X,Z,E) = data[i]
        X = np.array(X)
        Z = np.array(Z)
        Err = np.array(E)
        label = None
        marker = None
        linestyle = '-'
        if with_marker:
            marker = markers.next()
        if legend is not None:
            label = legend[i]
        plt.errorbar(X,Z,yerr=Err,color=colors[i],marker=marker,markersize=5,linestyle=linestyle,label=label)
    #plt.axvline(0.94050393,color='b',linestyle='--',linewidth=2)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    #plt.yscale('log')
    plt.legend(loc='best', shadow=True,prop={'size':14})
    plt.tight_layout()
    plt.grid()
    plt.savefig(saveto,dpi=300)
    plt.close()


def plotHist(data,legends,labels=['X','Y'],tick_labels=None,saveto='results',tick_interval=1):
    #plot histogram for energy spectrum
    plt.subplot(111)
    plt.figure(figsize=(8,6))
    dim = len(data[0])
    dimw = 0.35 #width of a pillar
    w = dimw * dim #width of a field
    x = np.arange(len(data))
    colors = matplotlib.cm.hsv(np.linspace(0.3,1,dim))
    for i in range(dim) :
        y = [d[i] for d in data]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.0,label=legends[i],color=colors[i])
    ticks = list(x[::tick_interval]+w/2)
    plt.xticks(ticks, fontsize = 14)
    #plt.gca().set_xticks(x+w/2)
    plt.gca().set_xticklabels(tick_labels)
    if len(tick_labels) != len(x[::tick_interval]):
        raise Exception('Length of ticks is not the same with lenght of labels')
    #plt.axvline(1.09457620/0.04,color='b',linestyle='--',linewidth=2)
    #plt.axvline(0.16839191/0.04,color='b',linestyle='--',linewidth=2)
    #plt.ylim([0,0.4])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(loc='best', shadow=True,prop={'size':14})
    plt.tight_layout()
    plt.savefig(saveto,dpi=300)
    plt.close()
    
                    



if __name__ == '__main__':
    

    #############number of residual defects number vs composition at a fixed number of cascades#########
    """
    x1 = [300, 700, 1000]
    yI210 = [7,4,2]
    yV210 = [638,397,216]
    x2 = [300, 700]
    yI310 = [20,9]
    yV310 = [913,486]
    data = [(x2,yV310),(x1,yV210),(x2,yI310),(x1,yI210)]
    legends = ['$\Sigma 5(310)-V$','$\Sigma 5(210)-V$','$\Sigma 5(310)-I$','$\Sigma 5(210)-I$']
    labels = [ 'Temperature (K)','Number of defect in the bulk']
    MultipleTwoVariable(data,saveto='defects_GB_pka500.png',labels=labels,legend=legends)
    """
   


    ### 3d bar plot of cluster size distribution after a number of pka ###
    """
    pka = [100,200,300,400,500]
    data = {}
    min_size = 10000
    max_size = 0
    GB = '0_1_2'
    path = '/media/mmjin/3A7AAE1E7AADD6C3/INLCluster08282017/GB_Cu/'+GB+'_GB/Consecutive_Cascades_300K/case1/'
    for i,item in enumerate(pka) :
        filename = path +'cluster_sizes_pka'+str(item)+'.txt'
        print filename
        if os.path.isfile(filename):
            data[item] = np.genfromtxt(filename)
            min_size = min(min_size,data[item].min())
            max_size = max(max_size,data[item].max())
    bins = 11
    min_size = 0
    max_size = 1100
    bin_edges = np.linspace(min_size,max_size,bins+1) 
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    xpos = range(1,bins+1) 
    ypos = range(1,len(pka)+1)
    colors = matplotlib.cm.hsv(np.linspace(0.2,1,len(pka)))
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111,projection='3d')
    dx = 0.4
    dy = 0.4
    for i,key in enumerate(pka) :
        (hist,_) = np.histogram(data[key],bins=bin_edges)
        print hist
        for j in range(len(hist)): 
            dz = hist[j]
            if dz > 1.0e-10:
                ax.bar3d(xpos[j],ypos[i],0,dx,dy,dz,color=colors[i],alpha=0.75)

    ax.set_xticks(np.array(xpos)+dx)
    ax.set_yticks(np.array(ypos)+0.2)

    ax.w_xaxis.set_ticklabels([str(int(bin_edges[i])) for i in range(1,bins+1)],fontsize = 14,fontweight='bold')
    ax.w_yaxis.set_ticklabels([str(key) for key in pka],fontsize = 14,fontweight='bold')
    ax.w_zaxis.set_ticklabels([str(int(i)) for i in ax.get_zticks()],fontsize = 14,fontweight='bold')
    ax.set_xlabel('Defective structure size',fontweight='bold')
    ax.set_ylabel('Number of cascades',fontweight='bold')
    ax.set_zlabel('Occurrence',fontweight='bold')
    #fig.autofmt_xdate()
    #ax.view_init(azim=120)
    plt.tight_layout()
    plt.savefig("cluster_size_dist_"+GB+".png",dpi=300)
    plt.close()
    """

    #############number of SFTs vs dpas at all spacings#########
    """
    cutoff_size = 13# only SFT with size > cutoff_size is counted
    cases = 3
    GBs = ['0_1_2_GB', '0_1_3_GB']
    GB = GBs[1]

    max_pka = 1300
    spacings = [3,6,9,12,15]
    real_spacings = 5.7047703443944663e+01*np.arange(1,len(spacings)+1)/10.0 #spacing in nm in simulations
    min_vol = (5.6669505990980426e+01*2) * (5.6669505990980426e+01*2) * (5.4307229155408955e+01*2) / 1000.0 #volume for smallest twin spacing in nm^3
    vols = np.arange(1,len(spacings)+1)*min_vol
    steps = [50,100,150,200,250]
    num_dpas = 5
    dpas = np.arange(1,num_dpas+1)*50*5000.0/2.0/30.0/117600 #NRT approx E/2Ed
    Z = np.zeros((len(spacings),num_dpas))
    for j,spacing in enumerate(spacings) : 
        for i,cur_pka in enumerate(np.arange(1,num_dpas+1)*steps[j]):
            effective_case = 0
            SFTs = 0
            for count in range(1,cases+1):
                filename = r'/home/mmjin/Dropbox (MIT)/Sharing/Grain_boundary_radiation_damage/Results/twin_spacing/'+str(spacing)+'nm/'+GB+'/cluster_sizes_pka'+str(cur_pka)+'_case'+str(count)+'.txt'
                if os.path.isfile(filename):
                    print filename
                    effective_case += 1
                    SFTs += np.sum(np.genfromtxt(filename)>cutoff_size)
            print 'Effective number of cases: ',effective_case
            print '\n'
            Z[i,j] = SFTs *1.0 /effective_case/vols[j]

    start_dpa = 1 #start from this dpa
    data = []
    legends = [str('%.2f dpa' %i) for i in dpas[start_dpa:]]
    labels = [ 'Twin spacing (nm)','Defect Entities/$nm^3$']
    for i in range(start_dpa,num_dpas):
        data.append((real_spacings,Z[i,:]))
    MultipleTwoVariable(data,saveto='SFTs_cascade_'+GB+'.png',labels=labels,legend=legends)
    np.savetxt("SFTs_map"+GB+".csv", Z, delimiter=" ")
    """




    ##############plot temperature evolution across 10 cascades for system 5.7 nm 300K cas1##########
    """
    df = pd.DataFrame()
    for i in range(1,11):
        data = pd.read_csv('revision_data/temp_data_case1/thermo_block_'+str(i)+'.txt',delimiter='\s+',header=None)
        df = pd.concat([df,data],ignore_index=True)
    df[5].plot(y=5,use_index=True)
    plt.xlabel('Timesteps',fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=14)
    plt.tight_layout()
    plt.savefig('temp_evolve.png',dpi=300)
    plt.close()
    """
         
    ##############plot temperature evolution across 250 cascades for system 5.7 nm 300K cas1 ##########
    df = pd.DataFrame()
    for i in range(1,251):
        data = pd.read_csv('revision_data/temp_data_complete_case1/thermo_block_'+str(i)+'.txt',delimiter='\s+',header=None)
        df = pd.concat([df,data],ignore_index=True)
    plotdata = df[5][30::31]
    plt.plot(range(1,len(plotdata)+1),plotdata)
    plt.xlabel('Cascades',fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=14)
    plt.tight_layout()
    plt.savefig('temp_evolve_complete.png',dpi=300)
    plt.close()
