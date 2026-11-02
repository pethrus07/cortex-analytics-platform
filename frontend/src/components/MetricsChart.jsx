/**
 * A série de telemetria: temperatura, vibração e pressão num gráfico só.
 * Recebe os eventos crus da API e apenas os prepara para o Recharts — não
 * busca nada e não agrega nada. Atenção ao que o eixo X realmente é: está no
 * comentário do `data` abaixo.
 */
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function MetricsChart({ events }) {
  // Inverte pra ficar em ordem cronológica e mapeia pro formato do gráfico.
  // Duas coisas a saber antes de ler esse gráfico: (1) o eixo X é a POSIÇÃO
  // na lista, não o horário — leituras com espaçamento diferente aparecem
  // igualmente espaçadas; (2) a API devolve os eventos de TODOS os ativos
  // ordenados por timestamp, então cada linha costura ativos diferentes.
  // Serve de panorama; não serve para ler a curva de um equipamento.
  const data = events
    .slice()
    .reverse()
    .map((e, i) => ({
      idx: i,
      temperatura: e.temperature,
      vibracao: e.vibration,
      pressao: e.pressure,
    }));

  return (
    <div className="bg-cortex-card rounded-xl p-5 border border-white/5 shadow-lg">
      <h2 className="text-sm font-semibold text-gray-200 mb-4">
        Telemetria dos sensores
      </h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="idx" stroke="#6b7280" tick={{ fontSize: 11 }} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#141b2e", border: "1px solid #334155" }} />
          <Legend />
          <Line type="monotone" dataKey="temperatura" stroke="#3b82f6" dot={false} />
          <Line type="monotone" dataKey="vibracao" stroke="#f59e0b" dot={false} />
          <Line type="monotone" dataKey="pressao" stroke="#22c55e" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
