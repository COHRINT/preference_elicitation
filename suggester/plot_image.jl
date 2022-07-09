using Images
# using Plots
using LaTeXStrings
FILEPATH = "./images/Image1_raw.png"
USER_INIT = "yellow"
USER_COLOR = "blue"
SUGGEST_COLOR = "green"


function plot_image(image,init_points,user_points, suggestions, denied, fname,save)
    img = load(image)
    legend = false
    plot(img,title = L"\textrm{Interaction}")
    if length(init_points)>0
        scatter!(init_points[1], init_points[2], color=USER_INIT,markersize = 5, label="init",legend =legend)
    end
    if length(user_points)>0
        scatter!(user_points[1], user_points[2], color=USER_COLOR, markersize = 5, label="user",legend = legend)
    end
    if length(denied)>0
        scatter!(denied[1], denied[2], color="red", label="denied", markersize = 5, legend = legend)
    end
    if length(suggestions)>0
        a = scatter!(suggestions[1], suggestions[2], color=SUGGEST_COLOR, markersize = 5, label="suggested",legend = legend)
    end
    # pt = plot(1:10, 1:10, label="A", legend=legend)
    # a.o[:legend](bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    display(a)
    if save
        savefig(a, fname)
    end
    return a
end



function extract_xy(points,points_data)
    u_points_x = []
    u_points_y = []
    for p in points
        x = points_data[parse(Int64,p)][1]
        y = points_data[parse(Int64,p)][2]
        push!(u_points_x,x)
        push!(u_points_y,y)
    end
    return u_points_x,u_points_y
end

function extract_vector(points,points_data)
    """Function extracts the feature vectors from a provided set of indices"""
    u_points = []
    for p in points
        x = points_data[parse(Int64,p)]
        push!(u_points,x)

    end
    return u_points
end

# x = [50, 100]
# y = [50, 100]

# u_points = [x,y]

# w = [100, 300]
# z = [200, 300]

# a_points = [w, z]

# plot_image(u_points, a_points, "data/test.png")