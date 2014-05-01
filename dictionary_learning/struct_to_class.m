function [ new_class_inst ] = struct_to_class ( new_struct_inst )
%struct_to_class  Takes a struct with a classname property and converts it
% back to a class of the named type.
    
    % If nothing else, we return the same thing we started with.
    new_class_inst = new_struct_inst;
    
    % We break groups of objects to deal with them separately.
    if (numel(new_class_inst) > 1)
        % Cells require {} for indexing. Otherwise, we don't extract their
        % contents. Everything else is the same.
        if iscell(new_class_inst)
            for i = 1:numel(new_class_inst)
                new_class_inst{i} = struct_to_class(new_class_inst{i});
            end
        else
            for i = 1:numel(new_class_inst)
                new_class_inst(i) = struct_to_class(new_class_inst(i));
            end
        end
    % We take objects and structs and deal with them the same way.
    elseif isstruct(new_class_inst) & isfield(new_class_inst, 'classname')
        % Take all fields and apply class_to_struct on them recursively.
        new_class_inst_fields = fieldnames(new_class_inst);
        for i = 1:numel(new_class_inst_fields)
            new_class_inst.(new_class_inst_fields{i}) = class_to_struct(new_class_inst.(new_class_inst_fields{i}));
        end
        
        % Grab the classname and drop it from the struct (maybe can drop
        % this).
        classname = new_class_inst.classname
        new_class_inst = rmfield(new_class_inst, 'classname')

        % Simply calls the class with the struct.
        new_class_inst = eval([classname '(new_class_inst)']);
    end
    
end

