struct NotYetImplemented <: Exception
    msg::String
    function NotYetImplemented(type::DataType, msg:stirng)
        this = new(msg)
        return this
    end
end
