import matplotlib.pyplot as plt

def show_images(images, labels, limit = 16, figsize=(20,4)):
    max_plot = len(images) if len(images) < limit else limit
    rows = 2
    cols=max_plot//rows
    
    fig = plt.figure(figsize=figsize)
    
    for i in range(max_plot):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(str(labels[i]))
        
    plt.show()
    