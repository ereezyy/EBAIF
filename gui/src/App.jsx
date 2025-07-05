import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  Zap, 
  Activity, 
  Users, 
  Settings, 
  Play, 
  Pause, 
  RotateCcw,
  TrendingUp,
  Network,
  Cpu,
  Target,
  Sparkles
} from 'lucide-react'

function App() {
  const [isRunning, setIsRunning] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState(null)
  const [agents, setAgents] = useState([
    { id: 1, name: 'Alpha', type: 'Neural', status: 'active', fitness: 0.87, connections: 12 },
    { id: 2, name: 'Beta', type: 'Evolutionary', status: 'learning', fitness: 0.76, connections: 8 },
    { id: 3, name: 'Gamma', type: 'Consensus', status: 'idle', fitness: 0.92, connections: 15 },
    { id: 4, name: 'Delta', type: 'Swarm', status: 'active', fitness: 0.68, connections: 6 },
  ])
  const [systemMetrics, setSystemMetrics] = useState({
    totalAgents: 4,
    activeAgents: 2,
    avgFitness: 0.81,
    totalConnections: 41,
    evolutionCycles: 1247,
    consensusRate: 0.94
  })

  useEffect(() => {
    const interval = setInterval(() => {
      if (isRunning) {
        setAgents(prevAgents => 
          prevAgents.map(agent => ({
            ...agent,
            fitness: Math.max(0, Math.min(1, agent.fitness + (Math.random() - 0.5) * 0.02)),
            connections: Math.max(0, agent.connections + Math.floor((Math.random() - 0.5) * 2))
          }))
        )
        
        setSystemMetrics(prev => ({
          ...prev,
          evolutionCycles: prev.evolutionCycles + 1,
          avgFitness: agents.reduce((sum, agent) => sum + agent.fitness, 0) / agents.length
        }))
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, agents])

  const toggleSystem = () => {
    setIsRunning(!isRunning)
  }

  const resetSystem = () => {
    setIsRunning(false)
    setAgents(prev => prev.map(agent => ({ ...agent, fitness: 0.5, connections: 5 })))
    setSystemMetrics(prev => ({ ...prev, evolutionCycles: 0, avgFitness: 0.5 }))
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/20'
      case 'learning': return 'text-yellow-400 bg-yellow-400/20'
      case 'idle': return 'text-gray-400 bg-gray-400/20'
      default: return 'text-gray-400 bg-gray-400/20'
    }
  }

  const getTypeIcon = (type) => {
    switch (type) {
      case 'Neural': return <Brain className="w-5 h-5" />
      case 'Evolutionary': return <TrendingUp className="w-5 h-5" />
      case 'Consensus': return <Users className="w-5 h-5" />
      case 'Swarm': return <Network className="w-5 h-5" />
      default: return <Cpu className="w-5 h-5" />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Sparkles className="w-8 h-8 text-neural-400" />
              <motion.div 
                className="absolute inset-0 w-8 h-8 text-neural-400"
                animate={{ rotate: isRunning ? 360 : 0 }}
                transition={{ duration: 2, repeat: isRunning ? Infinity : 0, ease: "linear" }}
              >
                <Sparkles className="w-8 h-8" />
              </motion.div>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-neural-400 to-purple-400 bg-clip-text text-transparent">
                EBAIF Control Center
              </h1>
              <p className="text-gray-400">Evolutionary Behavior AI Framework</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleSystem}
              className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all ${
                isRunning 
                  ? 'bg-red-500 hover:bg-red-600 text-white' 
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              <span>{isRunning ? 'Pause' : 'Start'}</span>
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={resetSystem}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl font-medium bg-gray-700 hover:bg-gray-600 text-white transition-all"
            >
              <RotateCcw className="w-5 h-5" />
              <span>Reset</span>
            </motion.button>
          </div>
        </motion.div>

        {/* System Metrics */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 mb-8"
        >
          {[
            { label: 'Total Agents', value: systemMetrics.totalAgents, icon: Users },
            { label: 'Active Agents', value: systemMetrics.activeAgents, icon: Activity },
            { label: 'Avg Fitness', value: systemMetrics.avgFitness.toFixed(3), icon: Target },
            { label: 'Connections', value: systemMetrics.totalConnections, icon: Network },
            { label: 'Evolution Cycles', value: systemMetrics.evolutionCycles, icon: TrendingUp },
            { label: 'Consensus Rate', value: systemMetrics.consensusRate.toFixed(2), icon: Zap },
          ].map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              className="agent-node p-4"
            >
              <div className="flex items-center space-x-3">
                <metric.icon className="w-6 h-6 text-neural-400" />
                <div>
                  <div className="text-2xl font-bold">{metric.value}</div>
                  <div className="text-sm text-gray-400">{metric.label}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Agent Grid */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {agents.map((agent, index) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedAgent(agent)}
              className="agent-node p-6 cursor-pointer hover:shadow-xl transition-all duration-300"
            >
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getTypeIcon(agent.type)}
                    <div>
                      <h3 className="font-bold text-lg">{agent.name}</h3>
                      <p className="text-sm text-gray-400">{agent.type}</p>
                    </div>
                  </div>
                  <motion.div 
                    className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}
                    animate={{ opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    {agent.status}
                  </motion.div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Fitness</span>
                    <span className="font-mono font-bold">{agent.fitness.toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div 
                      className="bg-gradient-to-r from-neural-500 to-purple-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${agent.fitness * 100}%` }}
                      transition={{ duration: 1, delay: index * 0.2 }}
                    />
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Connections</span>
                    <span className="font-mono font-bold">{agent.connections}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Agent Detail Modal */}
        <AnimatePresence>
          {selectedAgent && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6 z-50"
              onClick={() => setSelectedAgent(null)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="agent-node p-8 max-w-lg w-full"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="relative z-10">
                  <div className="flex items-center space-x-4 mb-6">
                    {getTypeIcon(selectedAgent.type)}
                    <div>
                      <h2 className="text-2xl font-bold">{selectedAgent.name}</h2>
                      <p className="text-gray-400">{selectedAgent.type} Agent</p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Status:</span>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedAgent.status)}`}>
                        {selectedAgent.status}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Fitness Score:</span>
                      <span className="font-mono font-bold">{selectedAgent.fitness.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Active Connections:</span>
                      <span className="font-mono font-bold">{selectedAgent.connections}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Learning Rate:</span>
                      <span className="font-mono font-bold">0.001</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Generation:</span>
                      <span className="font-mono font-bold">247</span>
                    </div>
                  </div>
                  
                  <div className="mt-6 pt-6 border-t border-gray-700">
                    <h3 className="font-semibold mb-3">Recent Activity</h3>
                    <div className="space-y-2 text-sm text-gray-400">
                      <div>• Evolved neural architecture at 14:32</div>
                      <div>• Formed new connection with Beta at 14:28</div>
                      <div>• Achieved fitness milestone at 14:25</div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default App